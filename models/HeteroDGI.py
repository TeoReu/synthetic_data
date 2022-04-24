import torch
from torch import nn
from torch_geometric.nn import GCNConv

from models.cat import CAT_DENSE, CAT_FIXED


class DoubleLayeredEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, type, depth):
        super().__init__()
        self.depth = depth
        self.conv = GCNConv(in_channels, hidden_channels, cached=False, normalize=True)
        self.prelu = nn.PReLU(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels, cached=False, normalize=True)
        self.prelu2 = nn.PReLU(hidden_channels)

        self.conv3 = GCNConv(hidden_channels, hidden_channels, cached=False, normalize=True)
        self.prelu3 = nn.PReLU(hidden_channels)
        if type == 'dense':
            self.cat = CAT_DENSE(hidden_channels)
        else:
            self.cat = CAT_FIXED()

    def forward(self, x, edge_index, edge_weight, edge_type):
        x = self.conv(x, edge_index, edge_weight)
        x = self.prelu(x)

        if self.depth >= 2:
            x = self.conv2(x, edge_index, edge_type.float())
            x = self.prelu2(x)

        if self.depth == 3:
            x = self.conv3(x, edge_index, edge_weight)
            x = self.prelu3(x)

        n1, n2 = torch.split(x, int(x.size(0) / 2))

        return self.cat(n1, n2)


def corruption(x, edge_index, edge_weight, edge_type):
    return x[torch.randperm(x.size(0))], edge_index, edge_weight, edge_type

