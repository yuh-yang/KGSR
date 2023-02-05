import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.utils import softmax as scatter_softmax
import math
from logging import getLogger

class HGT(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_users,
                 n_relations,
                node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(HGT, self).__init__()

        self.logger = getLogger()

        self.no_attn_convs = nn.ModuleList()
        self.n_relations = n_relations
        self.n_users = n_users
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        initializer = nn.init.xavier_uniform_
        relation_emb = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.relation_emb = nn.Parameter(relation_emb)  # [n_relations - 1, in_channel]

        self.W_Q = nn.Parameter(torch.Tensor(channel, channel))
        self.W_UI_1 = nn.Parameter(torch.Tensor(2*channel, channel))
        self.b_UI_1 = nn.Parameter(torch.Tensor(1, channel))
        self.W_UI_2 = nn.Parameter(torch.Tensor(channel, 1))
        self.b_UI_2 = nn.Parameter(torch.Tensor(1, channel))
        
        self.n_heads = 2
        self.d_k = channel // self.n_heads

        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_UI_1)
        nn.init.xavier_uniform_(self.b_UI_1)
        nn.init.xavier_uniform_(self.W_UI_2)
        nn.init.xavier_uniform_(self.b_UI_2)

        self.n_hops = n_hops

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout
    
    def non_attn_forward(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, relation_emb):

        n_entities = entity_emb.shape[0]

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = relation_emb[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        """user aggregate"""
        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        return entity_agg, user_agg
        
    def shared_layer_forward(self, user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, relation_emb):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index

        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)

        key = key * relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)

        edge_attn_score = (query * key).sum(dim=-1) / math.sqrt(self.d_k)

        relation_emb = relation_emb[edge_type - 1] # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * relation_emb  # [-1, channel]
        value = neigh_relation_emb.view(-1, self.n_heads, self.d_k)

        edge_attn_score = scatter_softmax(edge_attn_score, head)
        entity_agg = value * edge_attn_score.view(-1, self.n_heads, 1)
        entity_agg = entity_agg.view(-1, self.n_heads*self.d_k)
        # 因为前面有了attn weight, 所以这里不需要再mean, 改为sum
        entity_agg = scatter_sum(src=entity_agg, index=head, dim_size=n_entities, dim=0)

        item_agg = inter_edge_w.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        # w_attn = self.ui_weighting(user_emb, entity_emb, inter_edge)
        # item_agg += w_attn.unsqueeze(-1) * entity_emb[inter_edge[1, :]]
        user_agg = scatter_sum(src=item_agg, index=inter_edge[0, :], dim_size=user_emb.shape[0], dim=0)
        return entity_agg, user_agg


    # @TimeCounter.count_time(warmup_interval=4)
    def forward(self, user_emb, entity_emb, edge_index, edge_type,
                inter_edge, inter_edge_w, mess_dropout=True, dec_inter_edge=None, dec_inter_edge_w=None, dec_edge_index=None, dec_edge_type=None):

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        for i in range(self.n_hops):
            entity_emb, user_emb = self.shared_layer_forward(user_emb, entity_emb, edge_index, edge_type, inter_edge, inter_edge_w, self.relation_emb)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

            if i == self.n_hops - 1 and dec_edge_index is not None:
                entity_emb, user_emb = self.non_attn_forward(user_emb, entity_emb, dec_edge_index, dec_edge_type, inter_edge, inter_edge_w, self.relation_emb)
                """message dropout"""
                if mess_dropout:
                    entity_emb = self.dropout(entity_emb)
                    user_emb = self.dropout(user_emb)
                entity_emb = F.normalize(entity_emb)
                user_emb = F.normalize(user_emb)
                
                entity_res_emb = torch.add(entity_res_emb, entity_emb)
                user_res_emb = torch.add(user_res_emb, user_emb)

            elif i == self.n_hops - 1 and dec_inter_edge is not None:
                if dec_inter_edge_w is None:
                    dec_inter_edge_w = torch.ones_like(dec_inter_edge[0, :], device=user_emb.device).float()
                entity_emb, user_emb = self.non_attn_forward(user_emb, entity_emb, edge_index, edge_type, dec_inter_edge, dec_inter_edge_w, self.relation_emb)
                """message dropout"""
                if mess_dropout:
                    entity_emb = self.dropout(entity_emb)
                    user_emb = self.dropout(user_emb)
                entity_emb = F.normalize(entity_emb)
                user_emb = F.normalize(user_emb)
                
                entity_res_emb = torch.add(entity_res_emb, entity_emb)
                user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb
    
    def norm_attn_computer(self, entity_emb, edge_index, edge_type, print=False, return_logits=False):
        head, tail = edge_index

        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)

        key = key * self.relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)

        edge_attn = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
        edge_attn_logits = edge_attn.mean(-1).detach()
        # 按照head node进行softmax
        edge_attn_score = scatter_softmax(edge_attn_logits, head)
        # 除以head node的度进行归一化
        norm = scatter_sum(torch.ones_like(head), head, dim=0, dim_size=entity_emb.shape[0])
        norm = torch.index_select(norm, 0, head)
        edge_attn_score = edge_attn_score * norm
        # 打印attn score
        if print:
            # random_idx_print = np.random.randint(0, edge_attn_score.shape[0], 10)
            # self.logger.info("edge_attn_score: {}; std: {}".format(edge_attn_score[random_idx_print], edge_attn_score.std()))
            self.logger.info("edge_attn_score std: {}".format(edge_attn_score.std()))
        if return_logits:
            return edge_attn_score, edge_attn_logits
        return edge_attn_score
    
    def ui_weighting(self, user_emb, entity_emb, inter_edge):
        users = inter_edge[0, :]
        items = inter_edge[1, :]
        w = torch.cat([user_emb[users], entity_emb[items]], dim=-1) @ self.W_UI_1
        w = F.sigmoid(w + self.b_UI_1)
        w = torch.squeeze(w @ self.W_UI_2)
        # relax: learning to drop
        noise = torch.rand_like(w)
        noise = torch.log(noise) - torch.log(1 - noise)
        w = torch.sigmoid((w+noise) / 0.5)
        w = scatter_softmax(w, users)
        # re-construct interact_mat
        return w
