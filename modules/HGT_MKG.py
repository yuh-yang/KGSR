import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.utils import softmax as scatter_softmax
import math
from .HGCN import HGCNLayer


class HGTLayer(nn.Module):
    """
    Relational Path-aware Convolution Network
    """
    def __init__(self, channel, n_relations):
        super(HGTLayer, self).__init__()

        self.W_K = nn.Parameter(torch.Tensor(channel, channel))
        self.W_Q = nn.Parameter(torch.Tensor(channel, channel))
        self.W_V = nn.Parameter(torch.Tensor(channel, channel))
        self.W_O = nn.Parameter(torch.Tensor(channel, channel))

        self.n_heads = 4
        self.d_k = channel // self.n_heads

        self.relation_att = nn.Parameter(torch.Tensor(n_relations-1, self.n_heads, self.d_k, self.d_k)) # not include interact


        self.relation_msg = nn.Parameter(torch.Tensor(n_relations-1, self.n_heads, self.d_k, self.d_k))

        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_V)
        nn.init.xavier_uniform_(self.W_O)
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward_2(self, entity_emb, edge_index, edge_type, interact_mat, relation_emb):
        ''' no attention mapping '''
        n_entities = entity_emb.shape[0]
        head, tail = edge_index
        '''
            Step 1: Heterogeneous Mutual Attention
        '''
        query = entity_emb[head]
        key = entity_emb[tail]
        # N, 1, E * N, E, E
        key = torch.matmul(key.unsqueeze(1), self.relation_att_2[edge_type - 1]).squeeze()

        edge_attn_score = (query * key).sum(dim=-1) / math.sqrt(self.n_heads*self.d_k)

        ''' (3) 没有value mapping ***Current BEST*** '''
        # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        value = entity_emb[tail] * relation_emb[edge_type - 1]
        '''
            Softmax based on target node's id (edge_index_i).
        '''
        edge_attn_score = scatter_softmax(edge_attn_score, head)
        entity_agg = value * edge_attn_score.view(-1, 1)
        # 因为前面有了attn weight, 所以这里不需要再mean, 改为sum
        entity_agg = scatter_sum(src=entity_agg, index=head, dim_size=n_entities, dim=0)

        user_agg = torch.sparse.mm(interact_mat, entity_emb)  # [n_users, channel]
        return entity_agg, user_agg
    
    def forward_1(self, entity_emb, edge_index, edge_type, interact_mat, relation_emb):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index
        '''
            Step 1: Heterogeneous Mutual Attention
        '''
        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_K).view(-1, self.n_heads, self.d_k)

        relation_context = self.relation_att[edge_type - 1]
        # N, n_heads, 1, d_k * N, n_heads, d_k, d_k -> N, n_heads, 1, d_k
        key = torch.matmul(key.unsqueeze(2), relation_context).squeeze() # exclude interact, remap [1, n_relations) to [0, n_relations-1)

        ''' 按照已有的边计算attention '''
        edge_attn_score = (query * key).sum(dim=-1) / math.sqrt(self.d_k)

        '''
            Step 2: Heterogeneous Message Passing
        '''
        ''' (1) tail作为value '''
        # value = self.W_V(entity_emb[tail]).view(-1, self.n_heads, self.d_k)
        # relation_emb = relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k) # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        # # N, n_heads, d_k * N, n_heads, d_k -> 
        # entity_agg = value * relation_emb

        ''' (2) tail*relation 作为value '''
        # relation_emb = relation_emb[edge_type - 1] # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        # neigh_relation_emb = entity_emb[tail] * relation_emb  # [-1, channel]
        # value = self.W_V(neigh_relation_emb).view(-1, self.n_heads, self.d_k)

        ''' (3) 没有value mapping ***Current BEST*** '''
        # relation_emb = relation_emb[edge_type - 1] # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        # neigh_relation_emb = entity_emb[tail] * relation_emb  # [-1, channel]
        # value = neigh_relation_emb.view(-1, self.n_heads, self.d_k)

        ''' (4) layer-wise relation emb, matmul relation '''
        relation_emb = self.relation_msg[edge_type -1]
        value = (entity_emb[tail] @ self.W_V).view(-1, self.n_heads, self.d_k)
        value = torch.matmul(value.unsqueeze(2), relation_emb).squeeze()

        '''
            Softmax based on target node's id (edge_index_i).
        '''
        edge_attn_score = scatter_softmax(edge_attn_score, head)
        entity_agg = value * edge_attn_score.view(-1, self.n_heads, 1)
        entity_agg = entity_agg.view(-1, self.n_heads*self.d_k) @ self.W_O
        # 因为前面有了attn weight, 所以这里不需要再mean, 改为sum
        entity_agg = scatter_sum(src=entity_agg, index=head, dim_size=n_entities, dim=0)

        user_agg = torch.sparse.mm(interact_mat, entity_emb)  # [n_users, channel]
        return entity_agg, user_agg

        n_entities = entity_emb.shape[0]

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = relation_emb[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)
        # please interpret the above line as follows:
        # entity_agg = torch.zeros(n_entities, channel)
        # for i in range(n_entities):
        #     entity_agg[i] = torch.mean(neigh_relation_emb[head == i], dim=0)

        """user aggregate"""
        user_agg = torch.sparse.mm(interact_mat, entity_emb)  # [n_users, channel]

        return entity_agg, user_agg


class HGT(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_users,
                 n_relations, interact_mat,
                node_dropout_rate=0.5, mess_dropout_rate=0.1, save_topk_edges=False):
        super(HGT, self).__init__()

        self.no_attn_convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        initializer = nn.init.xavier_uniform_
        relation_emb = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.relation_emb = nn.Parameter(relation_emb)  # [n_relations - 1, in_channel]

        # self.W_K = nn.Parameter(torch.Tensor(channel, channel))
        self.W_Q = nn.Parameter(torch.Tensor(channel, channel))

        # self.E = nn.Parameter(torch.Tensor(interact_mat.shape[1], interact_mat.shape[1]//20))
        
        self.n_heads = 2
        self.d_k = channel // self.n_heads

        # self.relation_att = nn.Parameter(torch.Tensor(n_relations-1, self.n_heads, self.d_k, self.d_k)) # not include interact

        # nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_Q)
        # nn.init.xavier_uniform_(self.E)
        # nn.init.xavier_uniform_(self.relation_att)

        for i in range(n_hops):
            self.no_attn_convs.append(HGCNLayer())

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

        self.save_topk_edges = save_topk_edges
        if save_topk_edges:
            self.topk_edges = []
    
        
    def shared_layer_forward(self, entity_emb, edge_index, edge_type, interact_mat, relation_emb):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index

        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)

        # relation_context = self.relation_att[edge_type - 1]

        # key = torch.matmul(key.unsqueeze(2), relation_context).squeeze()

        key = key * relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)

        edge_attn_score = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
        if self.save_topk_edges:
            topk_attn_edges = torch.topk(edge_attn_score.sum(-1), 512, sorted=False)[1].cpu().numpy()
            self.topk_edges.append(topk_attn_edges)

        relation_emb = relation_emb[edge_type - 1] # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * relation_emb  # [-1, channel]
        value = neigh_relation_emb.view(-1, self.n_heads, self.d_k)

        edge_attn_score = scatter_softmax(edge_attn_score, head)
        entity_agg = value * edge_attn_score.view(-1, self.n_heads, 1)
        entity_agg = entity_agg.view(-1, self.n_heads*self.d_k)
        # 因为前面有了attn weight, 所以这里不需要再mean, 改为sum
        entity_agg = scatter_sum(src=entity_agg, index=head, dim_size=n_entities, dim=0)
        # n_users, n_entities x n_entities, channel
        user_agg = torch.sparse.mm(interact_mat, entity_emb)  # [n_users, channel]
        # user_agg = torch.mm(torch.sparse.mm(interact_mat, self.E), self.E.T @ entity_emb)
        return entity_agg, user_agg


    # @TimeCounter.count_time(warmup_interval=4)
    def forward(self, user_emb, entity_emb, edge_index, edge_type,
                interact_mat, mess_dropout=True):

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        if self.save_topk_edges:
            self.topk_edges = []
        for i in range(len(self.no_attn_convs)):
            entity_emb, user_emb = self.shared_layer_forward(entity_emb, edge_index, edge_type, interact_mat, self.relation_emb)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb
    
    def forward_hgcn_no_user(self, user_emb, entity_emb, edge_index, edge_type,
                interact_mat, mess_dropout=True):
        entity_res_emb = entity_emb  # [n_entity, channel]
        for i in range(len(self.no_attn_convs)):
            entity_emb = self.no_attn_convs[i].forward_no_user(entity_emb,
                                                 edge_index, edge_type, interact_mat,
                                                 self.relation_emb)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
            entity_emb = F.normalize(entity_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)

        return entity_res_emb
    
    def attn_computer(self, entity_emb, edge_index, edge_type, topk=None):
        head, tail = edge_index

        query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)

        # relation_context = self.relation_att[edge_type - 1]

        # key = torch.matmul(key.unsqueeze(2), relation_context).squeeze()

        key = key * self.relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)

        edge_attn_score = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
        edge_attn_score = edge_attn_score.sum(-1).detach()
        if topk is not None:
            return torch.topk(edge_attn_score, topk, sorted=False)
        return edge_attn_score
