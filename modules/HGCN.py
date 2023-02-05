import torch.nn as nn
import torch
from torch_scatter import scatter_mean
import torch.nn.functional as F


class HGCNLayer(nn.Module):
    """
    Relational Path-aware Convolution Network
    """
    def __init__(self):
        super(HGCNLayer, self).__init__()

    def forward(self, entity_emb, edge_index, edge_type, interact_mat, weight):

        n_entities = entity_emb.shape[0]

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)
        # please interpret the above line as follows:
        # entity_agg = torch.zeros(n_entities, channel)
        # for i in range(n_entities):
        #     entity_agg[i] = torch.mean(neigh_relation_emb[head == i], dim=0)

        """user aggregate"""
        user_agg = torch.sparse.mm(interact_mat, entity_emb)  # [n_users, channel]

        return entity_agg, user_agg

    def forward_no_user(self, entity_emb, edge_index, edge_type, interact_mat, weight):
        n_entities = entity_emb.shape[0]

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        return entity_agg

class HGCN(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_users,
                 n_relations, interact_mat,
                node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(HGCN, self).__init__()

        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        self.temperature = 0.2

        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

        for i in range(n_hops):
            self.convs.append(HGCNLayer())

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    # @TimeCounter.count_time(warmup_interval=4)
    def forward(self, user_emb, entity_emb, edge_index, edge_type,
                interact_mat, mess_dropout=True):

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](entity_emb,
                                                 edge_index, edge_type, interact_mat,
                                                 self.weight)

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

