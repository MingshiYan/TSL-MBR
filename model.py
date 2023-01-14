#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author:
# @Date  : 2021/11/1 16:16
# @Desc  :
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_set import DataSet
from gcn_conv import GCNConv
from utils import BPRLoss, EmbLoss


class GraphEncoder(nn.Module):
    def __init__(self, layers, hidden_dim, dropout):
        super(GraphEncoder, self).__init__()
        self.gnn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim, add_self_loops=False, cached=False) for i in range(layers)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x=x, edge_index=edge_index)
            # x = self.dropout(x)
        return x


class MRTCF(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(MRTCF, self).__init__()

        self.device = args.device
        self.layers = args.layers
        self.node_dropout = args.node_dropout
        self.message_dropout = nn.Dropout(p=args.message_dropout)
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.edge_index = dataset.edge_index
        self.all_edge_index = dataset.all_edge_index
        self.behaviors = args.behaviors
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        self.Graph_encoder = nn.ModuleDict({
            behavior: GraphEncoder(self.layers, self.embedding_size, self.node_dropout) for behavior in self.behaviors
        })
        self.global_graph_encoder = GraphEncoder(self.layers, self.embedding_size, self.node_dropout)
        self.user_MLP = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size),
            )
            for _ in range(len(self.behaviors) - 1)
        ])

        self.item_MLP = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size),
            )
            for _ in range(len(self.behaviors) - 1)
        ])

        self.fin_user_mlp = nn.Sequential(
            nn.Linear(self.embedding_size * len(self.behaviors), self.embedding_size * len(self.behaviors)),
        )

        self.fin_item_mlp = nn.Sequential(
            nn.Linear(self.embedding_size * len(self.behaviors), self.embedding_size * len(self.behaviors)),
        )

        self.reg_weight = args.reg_weight
        self.layers = args.layers
        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model

        self.storage_user_embeddings = None
        self.storage_item_embeddings = None

        self.apply(self._init_weights)

        self._load_model()

    def _init_weights(self, module):

        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False)

    def gcn_propagate(self, total_embeddings):
        """
        gcn propagate in each behavior
        """
        all_embeddings = []
        for behavior in self.behaviors:
            indices = self.edge_index[behavior].to(self.device)
            behavior_embeddings = self.Graph_encoder[behavior](total_embeddings, indices)
            behavior_embeddings = F.normalize(behavior_embeddings, dim=-1)
            all_embeddings.append(behavior_embeddings + total_embeddings)
        return all_embeddings

    def gcn(self, total_embeddings, indices):
        behavior_embeddings = self.global_graph_encoder(total_embeddings, indices.to(self.device))
        behavior_embeddings = F.normalize(behavior_embeddings, dim=-1)
        return total_embeddings + behavior_embeddings

    def forward(self, batch_data, epoch):
        self.storage_user_embeddings = None
        self.storage_item_embeddings = None

        self.user_embedding.weight.requires_grad = True
        self.item_embedding.weight.requires_grad = True

        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = self.gcn(all_embeddings, self.all_edge_index)
        all_embeddings = self.gcn_propagate(all_embeddings)
        total_loss = 0
        for index, behavior in enumerate(self.behaviors):
            data = batch_data[:, index]
            users = data[:, 0].long()
            items = data[:, 1:].long()
            user_all_embedding, item_all_embedding = torch.split(all_embeddings[index],
                                                                 [self.n_users + 1, self.n_items + 1])
            user_feature = user_all_embedding[users.view(-1, 1)]
            item_feature = item_all_embedding[items]
            scores = torch.sum(user_feature * item_feature, dim=2)
            if epoch % 2 == 0:
                total_loss += self.bpr_loss(scores[:, 0], scores[:, 1])
        total_loss = total_loss + self.reg_weight * self.emb_loss(self.user_embedding.weight,
                                                                  self.item_embedding.weight)
        if epoch % 2 > 0:
            self.user_embedding.weight.requires_grad = False
            self.item_embedding.weight.requires_grad = False

            all_user_feature, all_item_feature = [], []
            data = batch_data[:, -2]
            users = data[:, 0].long()
            items = data[:, 1:].long()
            for index in range(len(self.behaviors)):
                user_all_embedding, item_all_embedding = torch.split(all_embeddings[index],
                                                                        [self.n_users + 1, self.n_items + 1])
                user_feature = user_all_embedding[users.view(-1, 1)]
                item_feature = item_all_embedding[items]
                if index < len(self.behaviors) - 1:
                    user_feature = self.user_MLP[index](user_feature)
                    item_feature = self.item_MLP[index](item_feature)
                all_user_feature.append(user_feature)
                all_item_feature.append(item_feature)

            user_feature = torch.cat(all_user_feature, dim=-1)
            item_feature = torch.cat(all_item_feature, dim=-1)
            user_feature = self.fin_user_mlp(user_feature)
            item_feature = self.fin_item_mlp(item_feature)
            scores = torch.sum(user_feature * item_feature, dim=2)
            if epoch % 2 == 1:
                total_loss += self.bpr_loss(scores[:, 0], scores[:, 1])

        total_loss = total_loss + self.reg_weight * self.emb_loss(self.user_embedding.weight, self.item_embedding.weight)

        return total_loss

    def full_predict(self, users):
        if self.storage_user_embeddings is None:
            user_all, item_all = [], []
            all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
            all_embeddings = self.gcn(all_embeddings, self.all_edge_index)
            all_embeddings = self.gcn_propagate(all_embeddings)
            for index, behavior in enumerate(self.behaviors):
                tmp_user_embeddings, tmp_item_embeddings = torch.split(all_embeddings[index],
                                                                       [self.n_users + 1, self.n_items + 1])
                if behavior != 'buy':
                    tmp_user_embeddings = self.user_MLP[index](tmp_user_embeddings)
                    tmp_item_embeddings = self.item_MLP[index](tmp_item_embeddings)
                user_all.append(tmp_user_embeddings)
                item_all.append(tmp_item_embeddings)
            user_all = torch.cat(user_all, dim=-1)
            item_all = torch.cat(item_all, dim=-1)
            self.storage_user_embeddings = self.fin_user_mlp(user_all)
            self.storage_item_embeddings = self.fin_item_mlp(item_all)

        user_emb = self.storage_user_embeddings[users.long()]
        scores = torch.matmul(user_emb, self.storage_item_embeddings.transpose(0, 1))

        return scores

