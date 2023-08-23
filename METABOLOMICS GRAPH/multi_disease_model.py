import pandas as pd 
import numpy as np 
import os
import json
import random 
import math

import pingouin as pg
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torchmetrics
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU,SELU


from sklearn import preprocessing
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate, GridSearchCV, cross_val_score, RepeatedStratifiedKFold, StratifiedKFold

# Graph utility function
import torch_geometric.utils as pyg_utils
import torch_geometric.nn as pyg_nn
import torch_geometric.transforms as T
import torch_geometric

from torch_geometric.nn import TransformerConv,GPSConv, GATConv, TopKPooling, BatchNorm, GATv2Conv, GINConv,GCNConv, SAGEConv,MLP,GraphConv,SAGPooling
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, Set2Set
from torch_geometric.graphgym import BatchNorm1dNode
from torch_geometric.nn import MLP
import copy




class sharedGNNblock(nn.Module):
    def __init__(self, 
                GNNmodules, 
                feature_size,
                 hidden_channels,
                 dropout =0, 
                 norm_fn='nn.BatchNorm1d', 
                 activation='nn.SELU', 
                 aggr='add'  ):

        super().__init__()
        if norm_fn is not None and isinstance(norm_fn, str):
            m = norm_fn.split('.')
            self.norm_fn = getattr(nn, m[1])
            print(self.norm_fn)
        if activation is not None and isinstance(activation, str):
            m = activation.split('.')
            self.activation = getattr(nn, m[1])
            self.activation = self.activation()
            print(self.activation)
        self.dropout = nn.Dropout(p=dropout)


        ## shared GNN block 
        # GNN layers  
        self.n_GNNlayers = len(GNNmodules)
        self.GNNs = nn.ModuleList()
        if isinstance(GNNmodules,list):
            for i, module in enumerate(GNNmodules):
                if i == 0:
                    self.GNNs.append(self.getGNN(gnntype = module, in_channels= feature_size, out_channels = hidden_channels, aggr=aggr))
                else:
                    self.GNNs.append(self.getGNN(gnntype = module, in_channels= hidden_channels, out_channels = hidden_channels, aggr=aggr))

        # normalization layers 
        self.norms = nn.ModuleList()
        for _ in range(self.n_GNNlayers - 1):
            self.norms.append(copy.deepcopy(self.norm_fn(hidden_channels)))

        self.reset_parameters()

        self.pool_aggr = Linear(hidden_channels*3, hidden_channels)

    def forward(self, x, edge_index,batch_index):
        for i in range(self.n_GNNlayers-1):
            x = self.GNNs[i](x, edge_index)
            x = self.norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.GNNs[-1](x, edge_index)
        x1 = global_add_pool(x,batch_index)
        x2 = global_mean_pool(x, batch_index)
        x3 = global_max_pool(x, batch_index)
        x = self.pool_aggr(torch.cat((x1,x2,x3),axis=-1))

        return x 

    def getGNN(self,gnntype, in_channels,out_channels, aggr):
        model = getattr(pyg_nn, gnntype)
        if gnntype == 'GINConv':
            gnn = model(nn = nn.Linear(in_channels,out_channels ),train_eps=True)
        elif gnntype == 'GCNConv':
            gnn = model(in_channels, out_channels)
        elif gnntype == 'GraphConv':
            gnn = model(in_channels, out_channels, aggr = aggr)

        return gnn

    def reset_parameters(self):
        for layer in self.GNNs:
            layer.reset_parameters()

        for norm in self.norms:
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)

                   

class headBlocks(nn.Module):
    def __init__(self,in_channels,
                 hidden_channels,
                 out_channels,
                 n_layers,
                 task_names,
                 dropout =0, 
                 norm_fn='nn.BatchNorm1d',
                 activation='nn.SELU', aggr='add' ):
        super().__init__()

        if norm_fn is not None and isinstance(norm_fn, str):
            m = norm_fn.split('.')
            self.norm_fn = getattr(nn, m[1])
            print(self.norm_fn)
        if activation is not None and isinstance(activation, str):
            m = activation.split('.')
            self.activation = getattr(nn, m[1])
            self.activation = self.activation()
            print(self.activation)        

        self.heads = nn.ModuleDict()
        for task in task_names:
            self.heads[task] = nn.ModuleList([pyg_nn.models.MLP(in_channels=in_channels, hidden_channels=hidden_channels,out_channels=hidden_channels,num_layers = n_layers,
                                    dropout=dropout, act= self.activation, norm ='BatchNorm'),
                                pyg_nn.models.MLP(in_channels= 306, hidden_channels=hidden_channels,out_channels=hidden_channels,num_layers = n_layers,
                                    dropout=dropout, act= self.activation, norm ='BatchNorm'),
                                pyg_nn.models.MLP(in_channels= hidden_channels*2, hidden_channels=64,out_channels=out_channels,num_layers = 3,
                                    dropout=dropout, act= self.activation, norm ='BatchNorm')])


        self.reset_parameters()


    def forward(self,shared_out, emb):
        results = {task: layer[2](torch.cat((layer[0](shared_out),layer[1](emb)),axis=-1)) for task,layer in self.heads.items()}
        return results

    def reset_parameters(self):
        for h in self.heads:
            for m in h:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)




class MultiDiseaseNetwork(nn.Module):
    def __init__(self,
                GNNmodules, 
                feature_size,
                 hidden_channel1,
                 hidden_channel2,
                 out_channels,
                 n_layers,
                 task_names,
                 dropout =0, 
                 norm_fn='nn.BatchNorm1d', 
                 activation='nn.SELU', 
                 aggr='add'):
        super().__init__()

        
        self.sharedGNNblock = sharedGNNblock(GNNmodules=GNNmodules, feature_size= feature_size, hidden_channels=hidden_channel1, dropout=dropout, norm_fn = norm_fn, activation=activation,aggr=aggr)
        self.headBlocks = headBlocks(in_channels = hidden_channel1, hidden_channels=hidden_channel2, out_channels=out_channels, 
                                        n_layers=n_layers,task_names=task_names,dropout=dropout,norm_fn=norm_fn,activation=activation, aggr=aggr)

        self.feature_emb = Linear(feature_size, 1)
        # 306*hidden_channel1
    def forward(self, data):
        origin , edge_index, batch_index, edge_weight = data.x, data.edge_index, data.batch, data.edge_weight
        shared_out = self.sharedGNNblock(origin,edge_index, batch_index)
        #shared_out = shared_out.reshape(-1,306*hidden_channel1)
        emb = self.feature_emb(origin).reshape(-1, 306)
        
        out = self.headBlocks(shared_out, emb)
        # final outcome is a dictionary ex) {'T2D':[0.05, 0.95]}
        return out 


