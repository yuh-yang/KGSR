'''
Created on July 1, 2020
PyTorch Implementation of KGIN
@author: Tinglin Huang (tinglin.huang@zju.edu.cn)
'''
__author__ = "huangtinglin"

from cmath import isnan
from utils.timer import TimeCounter
import random
import numpy as np
import torch
import torch.nn as nn
from .HGT_MKG import HGT
from .HGCN import HGCN
from logging import getLogger


def _mae_edge_mask_adapt(edge_index, edge_type, mask_indices):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    remain_indices = np.arange(n_edges)
    remain_indices = np.delete(remain_indices, mask_indices)
    remain_edge_index = edge_index[:, remain_indices]
    remain_edge_type = edge_type[remain_indices]
    masked_edge_index = edge_index[:, mask_indices]
    masked_edge_type = edge_type[mask_indices]
    return remain_edge_index, remain_edge_type, masked_edge_index, masked_edge_type


def _mae_edge_mask_random(edge_index, edge_type, mask_num=512):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    rand_idx = torch.randperm(n_edges)
    edge_index = edge_index[:, rand_idx]
    edge_type = edge_type[rand_idx]
    remain_edge_index = edge_index[:, mask_num:]
    remain_edge_type = edge_type[mask_num:]
    masked_edge_index = edge_index[:, :mask_num]
    masked_edge_type = edge_type[:mask_num]
    return remain_edge_index, remain_edge_type, masked_edge_index, masked_edge_type

# @TimeCounter.count_time(warmup_interval=4)


def _edge_sampling(edge_index, edge_type, rate=0.5):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    random_indices = np.random.choice(
        n_edges, size=int(n_edges * rate), replace=False)
    return edge_index[:, random_indices], edge_type[random_indices]

# @TimeCounter.count_time(warmup_interval=4)


def _sparse_dropout(x, rate=0.5):
    noise_shape = x._nnz()

    random_tensor = rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).type(torch.bool)
    i = x._indices()
    v = x._values()

    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
    return out * (1. / (1 - rate))


class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat):
        super(Recommender, self).__init__()
        self.args_config = args_config
        self.logger = getLogger()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
            else torch.device("cpu")

        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(
            adj_mat).to(self.device)

        self.edge_index, self.edge_type = self._get_edges(graph)

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)

        self.gcn = HGT(channel=self.emb_size,
                       n_hops=self.context_hops,
                       n_users=self.n_users,
                       n_relations=self.n_relations,
                       interact_mat=self.interact_mat,
                       node_dropout_rate=self.node_dropout_rate,
                       mess_dropout_rate=self.mess_dropout_rate)

        self.print_shapes()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        """node dropout"""
        if self.node_dropout:
            edge_index, edge_type = _edge_sampling(
                self.edge_index, self.edge_type, self.node_dropout_rate)
            interact_mat = _sparse_dropout(
                self.interact_mat, self.node_dropout_rate)
        else:
            edge_index, edge_type = self.edge_index, self.edge_type
            interact_mat = self.interact_mat
        entity_gcn_emb, user_gcn_emb = self.gcn(user_emb,
                                                            item_emb,
                                                            edge_index,
                                                            edge_type,
                                                            interact_mat,
                                                            mess_dropout=self.mess_dropout)
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]

        loss, rec_loss, reg_loss = self.create_bpr_loss(u_e, pos_e, neg_e)

        if self.args_config.mae:
            mae_loss = self.forward_mae(
                user_emb, item_emb, edge_index, edge_type, interact_mat)
            return loss+mae_loss, rec_loss, reg_loss, mae_loss
        return loss, rec_loss, reg_loss

    def forward_mae(self, user_emb, item_emb, edge_index, edge_type, interact_mat):
        # topk_edges = np.unique(np.concatenate(self.gcn.topk_edges))
        # edge_index, edge_type, masked_edge_index, masked_edge_type = _mae_edge_mask_adapt(edge_index, edge_type, topk_edges)
        edge_index, edge_type, masked_edge_index, masked_edge_type = _mae_edge_mask_random(edge_index, edge_type)
        mae_entity_gcn_emb = self.gcn.forward_hgcn_no_user(user_emb,
                                                           item_emb,
                                                           edge_index,
                                                           edge_type,
                                                           interact_mat,
                                                           mess_dropout=self.mess_dropout)
        # batch_size, 2, channel
        node_pair_emb = mae_entity_gcn_emb[masked_edge_index.t()]
        # batch_size, channel
        masked_edge_emb = self.gcn.relation_emb[masked_edge_type-1]
        mae_loss = 0.1 * self.create_mae_loss(node_pair_emb, masked_edge_emb)
        return mae_loss

    def generate(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        return self.gcn(user_emb,
                        item_emb,
                        self.edge_index,
                        self.edge_type,
                        self.interact_mat,
                        mess_dropout=False)[:2]

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    # @TimeCounter.count_time(warmup_interval=4)
    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        if torch.isnan(mf_loss):
            self.logger.error("nan mf_loss")
            exit()

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def create_mae_loss(self, node_pair_emb, masked_edge_emb):
        head_embs, tail_embs = node_pair_emb[:, 0, :], node_pair_emb[:, 1, :]
        pos1 = tail_embs * masked_edge_emb
        # scores = (pos1 - head_embs).sum(dim=1).abs().mean(dim=0)
        scores = - \
            torch.log(torch.sigmoid(torch.mul(pos1, head_embs).sum(1))).mean()
        return scores

    def print_shapes(self):
        self.logger.info("########## Model Parameters ##########")
        self.logger.info("context_hops: %d", self.context_hops)
        self.logger.info("node_dropout: %d", self.node_dropout)
        self.logger.info("node_dropout_rate: %.1f", self.node_dropout_rate)
        self.logger.info("mess_dropout: %d", self.mess_dropout)
        self.logger.info("mess_dropout_rate: %.1f", self.mess_dropout_rate)
        self.logger.info('all_embed: {}'.format(self.all_embed.shape))
        self.logger.info('interact_mat: {}'.format(self.interact_mat.shape))
        self.logger.info('edge_index: {}'.format(self.edge_index.shape))
        self.logger.info('edge_type: {}'.format(self.edge_type.shape))
