'''
Created on July 1, 2020
PyTorch Implementation of KGIN
@author: Tinglin Huang (tinglin.huang@zju.edu.cn)
'''
__author__ = "huangtinglin"

from utils.timer import TimeCounter
import random
import numpy as np
import torch
import torch.nn as nn
from .HGT_CCL import HGT
from .grace import GRACE
from logging import getLogger
import math
import torch.nn.functional as F
from torch_scatter import scatter_sum
from scipy import sparse as sp

def _construct_ue_graph(ui_edge, ie_edge, n_users, n_items, n_entities):
    # ui_edge: [2, -1]
    # ie_edge: [2, -1]
    # return: [2, -1]
    ui_edge = ui_edge.cpu().numpy()
    ie_edge = ie_edge.cpu().numpy()
    # construct user-item graph
    ui_adj = sp.coo_matrix(
        (np.ones(ui_edge.shape[1]), (ui_edge[0], ui_edge[1])),
        shape=(n_users, n_entities))
    # construct item-entity graph
    ie_adj = sp.coo_matrix(
        (np.ones(ie_edge.shape[1]), (ie_edge[0], ie_edge[1])),
        shape=(n_entities, n_entities))
    # construct user-entity graph
    ue_adj = ui_adj.dot(ie_adj).tocoo()
    ue_edge = np.stack([ue_adj.row, ue_adj.col], axis=0)
    ue_edge = torch.from_numpy(ue_edge).long()

    return ue_edge

def _relation_aware_edge_sampling(edge_index, edge_type, n_relations, samp_rate=0.5):
    # exclude interaction
    for i in range(n_relations - 1):
        edge_index_i, edge_type_i = _edge_sampling(
            edge_index[:, edge_type == i], edge_type[edge_type == i], samp_rate)
        if i == 0:
            edge_index_sampled = edge_index_i
            edge_type_sampled = edge_type_i
        else:
            edge_index_sampled = torch.cat(
                [edge_index_sampled, edge_index_i], dim=1)
            edge_type_sampled = torch.cat(
                [edge_type_sampled, edge_type_i], dim=0)
    return edge_index_sampled, edge_type_sampled


def _mae_edge_mask_adapt_mixed(edge_index, edge_type, topk_egde_id):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    topk_egde_id = topk_egde_id.cpu().numpy()
    topk_mask = np.zeros(n_edges, dtype=np.bool)
    topk_mask[topk_egde_id] = True
    # add another group of random mask
    random_indices = np.random.choice(
        n_edges, size=topk_egde_id.shape[0], replace=False)
    random_mask = np.zeros(n_edges, dtype=np.bool)
    random_mask[random_indices] = True
    # combine two masks
    mask = topk_mask | random_mask

    remain_edge_index = edge_index[:, ~mask]
    remain_edge_type = edge_type[~mask]
    masked_edge_index = edge_index[:, mask]
    masked_edge_type = edge_type[mask]

    return remain_edge_index, remain_edge_type, masked_edge_index, masked_edge_type, mask


def _edge_sampling(edge_index, edge_type, samp_rate=0.5):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    random_indices = np.random.choice(
        n_edges, size=int(n_edges * samp_rate), replace=False)
    return edge_index[:, random_indices], edge_type[random_indices]


def _sparse_dropout(i, v=None, keep_rate=0.5):
    noise_shape = i.shape[1]

    random_tensor = keep_rate
    # the drop rate is 1 - keep_rate
    random_tensor += torch.rand(noise_shape).to(i.device)
    dropout_mask = torch.floor(random_tensor).type(torch.bool)

    i = i[:, dropout_mask]
    if v is not None:
        v = v[dropout_mask] / keep_rate
        return i, v
    else:
        return i


class MKG_UE(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat):
        super(MKG_UE, self).__init__()
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

        if args_config.dataset == 'last-fm':
            self.mae_coef = 0.1
            self.cl_coef = 0.01
        elif args_config.dataset == 'ml':
            self.mae_coef = 0.05
        elif args_config.dataset == 'mind-f':
            self.mae_coef = 0.1
            self.cl_coef = 0.001
            self.ue_coef = 0.05

        self.inter_edge, self.inter_edge_w = self._convert_sp_mat_to_tensor(
            adj_mat)

        self.edge_index, self.edge_type = self._get_edges(graph)

        self.ue_edge = _construct_ue_graph(self.inter_edge, self.edge_index, self.n_users, self.n_items, self.n_entities).to(self.device)
        self.ue_mlp = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size, bias=True),
        )

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)

        self.gcn = HGT(channel=self.emb_size,
                       n_hops=self.context_hops,
                       n_users=self.n_users,
                       n_relations=self.n_relations,
                       node_dropout_rate=self.node_dropout_rate,
                       mess_dropout_rate=self.mess_dropout_rate)

        self.print_shapes()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))

    def _convert_sp_mat_to_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return i.to(self.device), v.to(self.device)

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
        epoch_start = batch['batch_start'] == 0

        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        """node dropout"""
        # 1. 先按关系进行平衡采样;
        edge_index, edge_type = _relation_aware_edge_sampling(
            self.edge_index, self.edge_type, self.n_relations, self.node_dropout_rate)
        # 2. 然后按照attn+random进行mask，用来reconstruct;
        edge_attn_score, edge_attn_logits = self.gcn.norm_attn_computer(
            item_emb, edge_index, edge_type, print=epoch_start, return_logits=True)
        # for adaptive UI MAE
        item_attn_sumed = (scatter_sum(edge_attn_logits, edge_index[0], dim=0, dim_size=self.n_entities) + scatter_sum(
            edge_attn_logits, edge_index[1], dim=0, dim_size=self.n_entities))[:self.n_items]
        # for adaptive MAE training
        std = torch.std(edge_attn_score).detach()
        noise = -torch.log(-torch.log(torch.rand_like(edge_attn_score)))
        edge_attn_score = edge_attn_score + noise
        topk_v, topk_attn_edge_id = torch.topk(
            edge_attn_score, 256, sorted=False)
        top_attn_edge_type = edge_type[topk_attn_edge_id]
        enc_edge_index, enc_edge_type, masked_edge_index, masked_edge_type, mask_bool = _mae_edge_mask_adapt_mixed(
            edge_index, edge_type, topk_attn_edge_id)

        inter_edge, inter_edge_w = _sparse_dropout(
            self.inter_edge, self.inter_edge_w, self.node_dropout_rate)

        # rec task
        entity_gcn_emb, user_gcn_emb = self.gcn(user_emb,
                                                item_emb,
                                                enc_edge_index,
                                                enc_edge_type,
                                                inter_edge,
                                                inter_edge_w,
                                                mess_dropout=self.mess_dropout,
                                                )
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]

        loss, rec_loss, reg_loss = self.create_bpr_loss(u_e, pos_e, neg_e)

        # MAE task
        # mask_size, 2, channel
        node_pair_emb = entity_gcn_emb[masked_edge_index.t()]
        # mask_size, channel
        masked_edge_emb = self.gcn.relation_emb[masked_edge_type-1]
        # TODO: 这里能不能再让一个单层的HGT去解码？
        mae_loss = self.mae_coef * \
            self.create_mae_loss(node_pair_emb, masked_edge_emb)
        
        # UE MAE task
        ue_edge = _sparse_dropout(self.ue_edge, keep_rate=0.1)
        ue_edge_attn_score = self.gcn.ue_norm_attn_computer(user_emb, item_emb, ue_edge)
        noise = -torch.log(-torch.log(torch.rand_like(ue_edge_attn_score)))
        ue_edge_attn_score = ue_edge_attn_score + noise
        _, topk_ue_attn_edge_id = torch.topk(ue_edge_attn_score, 64, sorted=False)
        ue_attn_edge = ue_edge[:, topk_ue_attn_edge_id]
        random_e = self.n_items + torch.randint_like(ue_attn_edge[0], 0, self.n_entities - self.n_items)
        ue_u = self.ue_mlp(user_gcn_emb[ue_attn_edge[0,:]])
        ue_e = self.ue_mlp(entity_gcn_emb[ue_attn_edge[1,:]])
        neg_e = self.ue_mlp(entity_gcn_emb[random_e])

        # pos_scores = torch.sum(torch.mul(ue_u, ue_e), axis=1)
        # neg_scores = torch.sum(torch.mul(ue_u, neg_e), axis=1)
        # ue_loss = self.ue_coef * -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        ue_loss = self.ue_coef * -torch.log(torch.sigmoid(torch.mul(ue_u, ue_e).sum(1))).mean()

        loss_dict = {
            "rec_loss": loss.item(),
            "mae_loss": mae_loss.item(),
            "ue_loss": ue_loss.item(),
        }
        return loss + mae_loss + ue_loss, loss_dict

    def calc_topk_attn_edge(self, entity_emb, edge_index, edge_type, k):
        edge_attn_score = self.gcn.norm_attn_computer(
            entity_emb, edge_index, edge_type, return_logits=True)
        positive_mask = edge_attn_score > 0
        edge_attn_score = edge_attn_score[positive_mask]
        edge_index = edge_index[:, positive_mask]
        edge_type = edge_type[positive_mask]
        topk_values, topk_indices = torch.topk(
            edge_attn_score, k, sorted=False)
        return edge_index[:, topk_indices], edge_type[topk_indices]

    def generate(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        return self.gcn(user_emb,
                        item_emb,
                        self.edge_index,
                        self.edge_type,
                        self.inter_edge,
                        self.inter_edge_w,
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
            raise ValueError("nan mf_loss")

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

    def create_mae_loss(self, node_pair_emb, masked_edge_emb=None):
        head_embs, tail_embs = node_pair_emb[:, 0, :], node_pair_emb[:, 1, :]
        if masked_edge_emb is not None:
            pos1 = tail_embs * masked_edge_emb
        else:
            pos1 = tail_embs
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
        self.logger.info('interact_mat: {}'.format(self.inter_edge.shape))
        self.logger.info('edge_index: {}'.format(self.edge_index.shape))
        self.logger.info('edge_type: {}'.format(self.edge_type.shape))
