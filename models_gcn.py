import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops

# StaGr + GraphSplit
class GraphConvLayer_StaGr(torch.nn.Module):
    def __init__(self, in_channels, out_channels, norm):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.norm = norm

    def forward(self, x):
        # Apply linear transformation
        x = self.linear(x)
        out = torch.matmul(self.norm, x)

        return out

class GCN_StaGr(torch.nn.Module):
    def __init__(self, num_features, num_classes, x, edge_index):
        super().__init__()

        # Step 1: Form the adjacency matrix
        num_nodes = x.size(0)
        adjacency = torch.zeros(num_nodes, num_nodes, device=x.device)

        # Fill the adjacency matrix with edge connections
        adjacency[edge_index[0], edge_index[1]] = 1
        
        # Step 2: Add self-loops
        adjacency += torch.eye(num_nodes)

        # Step 3: Compute degree matrix
        degree = adjacency.sum(dim=1, keepdim=True)

        # Step 4: Normalize the adjacency matrix
        norm = torch.where(degree > 0, degree.pow(-0.5), torch.zeros_like(degree))
        norm = norm * norm.t()
        norm = norm * adjacency

        ## Layers
        self.gcn1 = GraphConvLayer_v4(num_features, 64, norm)
        self.gcn2 = GraphConvLayer_v4(64, num_classes, norm)

    def forward(self, x):
        h = self.gcn1(x).relu()
        z = self.gcn2(h)
        return F.log_softmax(z, dim=1)
    

## GrAd + NodePad
class GraphConvLayer_GrAd_NodePad(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
        # self.norm = norm

    def forward(self, x, norm):
        # Apply linear transformation
        x = self.linear(x)
        out = torch.matmul(norm, x)

        return out

class GCN_GrAd_NodePad(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        ## Layers
        self.gcn1 = GraphConvLayer_v6(num_features, 64)
        self.gcn2 = GraphConvLayer_v6(64, num_classes)

    def forward(self, x, norm):
        h = self.gcn1(x, norm).relu()
        z = self.gcn2(h, norm)

        return torch.log(F.softmax(z, dim=1))
    