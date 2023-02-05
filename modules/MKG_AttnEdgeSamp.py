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
from .HGT_AES import HGT
from logging import getLogger
import math
import torch.nn.functional as F
from torch_scatter import scatter_sum
from scipy import sparse as sp


def _generate_virtual_edges(n_entities, top_edge_types, n_virtual_edges=512):
    top_edge_types = np.random.permutation(top_edge_types.unique().cpu().numpy())[:4]
    # sample 512 virtual edge types from top 4 edge types
    virtual_edge_type = torch.from_numpy(np.random.choice(
        top_edge_types, size=n_virtual_edges, replace=True))
    virtual_edge_index = torch.randint(
        0, n_entities, (2, n_virtual_edges), dtype=torch.long)
    return virtual_edge_index, virtual_edge_type

def _attentive_edge_sampling(edge_index, edge_type, attn_scores, rate=0.8):
    # apply softmax to attention scores across all edges
    attn_scores = torch.softmax(attn_scores, dim=0)
    # sample edges according to attention scores
    n_edges = edge_index.shape[1]
    sampled_indices = np.random.choice(
        n_edges, size=int(n_edges * rate), replace=False, p=attn_scores.cpu().numpy())
    return edge_index[:, sampled_indices], edge_type[sampled_indices]

def _relation_aware_edge_sampling(edge_index, edge_type, n_relations, rate=0.5):
    # exclude interaction
    for i in range(n_relations - 1):
        edge_index_i, edge_type_i = _edge_sampling(
            edge_index[:, edge_type == i], edge_type[edge_type == i], rate)
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
    ## add another group of random mask
    random_indices = np.random.choice(n_edges, size=topk_egde_id.shape[0], replace=False)
    random_mask = np.zeros(n_edges, dtype=np.bool)
    random_mask[random_indices] = True
    ## combine two masks
    mask = topk_mask | random_mask

    remain_edge_index = edge_index[:, ~mask]
    remain_edge_type = edge_type[~mask]
    masked_edge_index = edge_index[:, mask]
    masked_edge_type = edge_type[mask]

    return remain_edge_index, remain_edge_type, masked_edge_index, masked_edge_type, mask

def _mae_ui_mask_adapt(edge_index, w, top_item_id):
    # """ 二阶采样 """
    # masked_nodes = [top_item_id]
    # for i in range(2):
    #     seeds = top_item_id if i == 0 else next_seeds
    #     next_seeds = list()
    # for seed in seeds:
    #     row_hit = (edge_index[0] == seed)
    #     col_hit = (edge_index[1] == seed)
    #     idct = row_hit | col_hit

    #     if i != 1:
    #         masked_rows = rows[idct]
    #         masked_cols = cols[idct]
    #         next_seeds.append(masked_rows)
    #         next_seeds.append(masked_cols)

    #     rows = rows[~idct]
    #     cols = cols[~idct]
    # if len(next_seeds) > 0:
    #     next_seeds = torch.unique(torch.concat(next_seeds))
    #     masked_nodes.append(next_seeds)
    mask = torch.zeros(w.shape[0], dtype=torch.bool).to(w.device)
    for item_id in top_item_id:
        mask = mask | (edge_index[1] == item_id)

    remain_w = w[~mask]
    masked_w = w[mask]
    remain_edge = edge_index[:, ~mask]
    masked_edge = edge_index[:, mask]
    return remain_edge, remain_w, masked_edge, masked_w, mask

def _mae_edge_mask_random(edge_index, edge_type, mask_num):
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

def _edge_sampling(edge_index, edge_type, rate=0.5):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    random_indices = np.random.choice(
        n_edges, size=int(n_edges * rate), replace=False)
    return edge_index[:, random_indices], edge_type[random_indices]

def _sparse_dropout(i, v, rate=0.5):
    noise_shape = i.shape[1]

    random_tensor = rate
    random_tensor += torch.rand(noise_shape).to(i.device)
    dropout_mask = torch.floor(random_tensor).type(torch.bool)

    i = i[:, dropout_mask]
    v = v[dropout_mask] * (1. / (1 - rate))

    return i, v


class MKG_AttnEdgeSamp(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat):
        super(MKG_AttnEdgeSamp, self).__init__()
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
        elif args_config.dataset == 'ml':
            self.mae_coef = 0.05

        self.inter_edge, self.inter_edge_w = self._convert_sp_mat_to_tensor(adj_mat)

        self.edge_index, self.edge_type = self._get_edges(graph)

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
        edge_attn_score, edge_attn_logits = self.gcn.norm_attn_computer(item_emb, edge_index, edge_type, print=epoch_start, return_logits=True)
        # for adaptive UI MAE
        edge_attn_sum_by_items = (scatter_sum(edge_attn_logits, edge_index[0], dim=0, dim_size=self.n_entities) + scatter_sum(edge_attn_logits, edge_index[1], dim=0, dim_size=self.n_entities))[:self.n_items]
        _, top_attn_items = torch.topk(edge_attn_sum_by_items, 100, dim=0)
        # for adaptive MAE training
        std = torch.std(edge_attn_score).detach()
        noise = -torch.log(-torch.log(torch.rand_like(edge_attn_score)))
        edge_attn_score = edge_attn_score + noise
        topk_v, topk_attn_edge_id = torch.topk(edge_attn_score, 256, sorted=False)
        top_attn_edge_type = edge_type[topk_attn_edge_id]
        enc_edge_index, enc_edge_type, masked_edge_index, masked_edge_type, mask_bool = _mae_edge_mask_adapt_mixed(edge_index, edge_type, topk_attn_edge_id)
        """
        # 3. 再从这个图里去掉20%注意力最低的边;
        edge_index, edge_type = _attentive_edge_sampling(edge_index, edge_type, edge_attn_score[~mask_bool], 0.8)
        """
        inter_edge, inter_edge_w = _sparse_dropout(
            self.inter_edge, self.inter_edge_w, self.node_dropout_rate)
        # UI MAE
        enc_inter_edge, enc_inter_edge_w, masked_inter_edge, masked_inter_edge_w, ui_mask_bool = _mae_ui_mask_adapt(inter_edge, inter_edge_w, top_attn_items)
        # inter_edge, inter_edge_w, masked_inter_edge, masked_inter_edge_w = _mae_edge_mask_random(inter_edge, inter_edge_w, 256)
        # rec task
        entity_gcn_emb, user_gcn_emb = self.gcn(user_emb,
                                                item_emb,
                                                enc_edge_index,
                                                enc_edge_type,
                                                enc_inter_edge,
                                                enc_inter_edge_w,
                                                mess_dropout=self.mess_dropout,
                                                # dec_inter_edge=inter_edge,
                                                # dec_inter_edge_w=inter_edge_w,
                                                dec_edge_index=edge_index,
                                                dec_edge_type=edge_type,)
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]

        loss, rec_loss, reg_loss = self.create_bpr_loss(u_e, pos_e, neg_e)

        # MAE task
        # mask_size, 2, channel
        node_pair_emb = entity_gcn_emb[masked_edge_index.t()]
        # mask_size, channel
        masked_edge_emb = self.gcn.relation_emb[masked_edge_type-1]
        # TODO: 这里能不能再让一个单层的HGT去解码？
        mae_loss = self.mae_coef * self.create_mae_loss(node_pair_emb, masked_edge_emb)

        # ui_pair_emb = torch.stack((user_gcn_emb[masked_inter_edge[0,:]], entity_gcn_emb[masked_inter_edge[1,:]]), dim=1)
        # mae_loss_ui = 0.1 * self.create_mae_loss(ui_pair_emb)

        """
        # 从top attn edge_type&id set中随机加入不存在的边 取64个最高的进行复原
        v_edge_index, v_edge_type = _generate_virtual_edges(self.n_entities, top_attn_edge_type, 512)
        v_edge_index, v_edge_type = self.calc_topk_attn_edge(item_emb, v_edge_index, v_edge_type, k=64)
        # weight is decided by std, as std goes higher, weight goes higher
        vmae_loss = 0.01 * self.create_mae_loss(entity_gcn_emb[v_edge_index.t()], self.gcn.relation_emb[v_edge_type-1])
        vmae_loss = vmae_loss * torch.exp(-1 / std)
        """
        # kg_loss = 0.01 * self.create_kg_loss(edge_index, edge_type, item_emb)
        # wa_loss = 1e-2 * self.create_wa_loss(user_emb, item_emb)
        loss_dict = {
            "rec_loss": loss.item(),
            "mae_loss": mae_loss.item(),
            # "mae_loss_ui": mae_loss_ui.item(),
            # "vmae_loss": vmae_loss.item(),
            # "wa_loss": wa_loss.item(),
            # "kg_loss": kg_loss.item(),
        }
        return loss + mae_loss, loss_dict
    
    def calc_topk_attn_edge(self, entity_emb, edge_index, edge_type, k):
        edge_attn_score = self.gcn.norm_attn_computer(entity_emb, edge_index, edge_type, return_logits=True)
        positive_mask = edge_attn_score > 0
        edge_attn_score = edge_attn_score[positive_mask]
        edge_index = edge_index[:, positive_mask]
        edge_type = edge_type[positive_mask]
        topk_values, topk_indices = torch.topk(edge_attn_score, k, sorted=False)
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
        scores = -torch.log(torch.sigmoid(torch.mul(pos1, head_embs).sum(1))).mean()
        return scores
    
    def create_kg_loss(self, edge_index, edge_type, entity_emb):
        def score_func(h, r, t):
            return torch.sum(h * r * t, dim=1)
        head, tail = edge_index
        head_emb, tail_emb = entity_emb[head], entity_emb[tail]
        edge_emb = self.gcn.relation_emb[edge_type-1]
        # random sample negative negs
        neg_tail = torch.randint(0, self.n_entities, (head.shape[0],))
        neg_tail_emb = entity_emb[neg_tail]
        loss = -torch.log(torch.sigmoid(score_func(head_emb, edge_emb, tail_emb) - score_func(head_emb, edge_emb, neg_tail_emb))).mean()
        return loss
    
    def create_wa_loss(self, user_emb, item_emb, tau=1.0):
        f = lambda x: torch.exp(x / tau)
        user_samp = torch.randint(0, self.n_users, (int(0.1*user_emb.shape[0]),))
        item_samp = torch.randint(0, self.n_items, (int(0.1*item_emb.shape[0]),))
        x = F.normalize(torch.cat([user_emb[user_samp], item_emb[item_samp]], dim=0))
        logits = torch.mm(x, x.t())
        label = torch.range(0, x.shape[0]-1).long().to(x.device)
        loss = F.cross_entropy(logits, label)
        return loss

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
