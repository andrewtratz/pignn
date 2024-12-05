import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, GELU, Tanh
from torch_geometric.nn import GCNConv, GATConv

from config import *


class GATSkip_WeightShare(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = eval(activation)()
        self.std_layer = GATConv(NODES, NODES, edge_dim=2) #nn.ModuleList([GATConv(NODES, NODES, edge_dim=2) for _ in range(4)])
        self.skip_layer = GATConv(NODES*2, NODES, edge_dim=2) 

    def forward(self, data):
        x = data['x']
        edge_index = data['edge_index']
        edge_attr = data['edge_attr']
        in_x = x.clone()
        
        for _ in range(LAYER_REPEATS):
            x = self.std_layer(x, edge_index.to(torch.int64), edge_attr)
            x = self.activation(x)

        x = self.skip_layer(torch.cat((in_x, x), axis=1), edge_index, edge_attr)
        return x
    

class GATSkip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = eval(activation)()
        self.std_layers = nn.ModuleList([GATConv(NODES, NODES, edge_dim=2) for _ in range(LAYER_REPEATS)])
        self.skip_layer = GATConv(NODES*2, NODES, edge_dim=2) 

    def forward(self, data):
        x = data['x']
        edge_index = data['edge_index']
        edge_attr = data['edge_attr']
        in_x = x.clone()
        
        for layer in self.std_layers:
            x = layer(x, edge_index, edge_attr)
            x = self.activation(x)

        x = self.skip_layer(torch.cat((in_x, x), axis=1), edge_index)
        return x



class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = Linear(FEATS, NODES)
        self.activation = eval(activation)()
        # self.conv_layers = nn.ModuleList([GCNConv(NODES, NODES)]*CONV_LAYERS)
        # self.conv_layers = nn.ModuleList([GATConv(NODES, NODES, edge_dim=2) for _ in range(CONV_LAYERS)])
        self.conv_layers = nn.ModuleList([GATSkip_WeightShare() for _ in range(CONV_LAYERS)])
        # self.conv_layers = nn.ModuleList([GATSkip() for _ in range(CONV_LAYERS)])
        self.final = GATConv(NODES, OUTPUTS, edge_dim=2)
        # self.final = GATConv(NODES, OUTPUTS, edge_dim=2)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.lin1(x)
        x = self.activation(x)
        for layer in self.conv_layers:
            x = layer({'x': x, 'edge_index': edge_index.to(torch.int64), 'edge_attr': edge_attr})
            x = self.activation(x)
        x = self.final(x, edge_index.to(torch.int64), edge_attr)

        return x
    

# BATCH_SIZE = 4

# FEATS = 9
# NODES = 16
# OUTPUTS = 4

# CONV_LAYERS = 16

# activation = 'GELU'

# class GATSkip(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.activation = eval(activation)()
#         self.std_layers = nn.ModuleList([GATConv(NODES, NODES, edge_dim=2) for _ in range(LAYER_REPEATS)])
#         self.skip_layer = GATConv(NODES*2, NODES, edge_dim=2) 

#     def forward(self, data):
#         x = data['x']
#         edge_index = data['edge_index']
#         edge_attr = data['edge_attr'].to(torch.int64)
#         in_x = x.clone()
#         for layer in self.std_layers:
#             x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
#             x = self.activation(x)

#         x = self.skip_layer(torch.cat((in_x, x), axis=1), edge_index)
#         return x

# class GCN(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lin1 = Linear(FEATS, NODES)
#         self.activation = eval(activation)()
#         # self.conv_layers = nn.ModuleList([GCNConv(NODES, NODES)]*CONV_LAYERS)
#         # self.conv_layers = nn.ModuleList([GATConv(NODES, NODES, edge_dim=2) for _ in range(CONV_LAYERS)])
#         self.conv_layers = nn.ModuleList([GATSkip() for _ in range(CONV_LAYERS)])
#         self.final = GATConv(NODES, OUTPUTS, edge_dim=2)
#         # self.final = GATConv(NODES, OUTPUTS, edge_dim=2)

#     def forward(self, data):
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

#         x = self.lin1(x)
#         x = self.activation(x)
#         for layer in self.conv_layers:
#             x = layer({'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr})
#             x = self.activation(x)
#         x = self.final(x, edge_index, edge_attr)

#         return x