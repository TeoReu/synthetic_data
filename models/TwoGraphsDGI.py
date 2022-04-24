import torch
import torch.nn as nn
from torch_geometric.nn import GCN, GCNConv

from models.cat import CAT_DENSE, CAT_FIXED
from models.dgi import AvgReadout, Discriminator


def corruption(x1, edge_index1, edge_weight1, x2, edge_index2, edge_weight2):
    rand = torch.randperm(x1.size(0))
    return x1[rand], edge_index1, edge_weight1, x2[rand], edge_index2, edge_weight2


class Encoder(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, hidden_channels, type, depth):
        super().__init__()
        self.depth = depth
        self.conv1 = GCNConv(in_channels_1, int(hidden_channels), cached=False, normalize=True)
        self.conv2 = GCNConv(in_channels_2, int(hidden_channels), cached=False, normalize=True)
        self.conv1a = GCNConv(int(hidden_channels), int(hidden_channels), cached=False, normalize=True)
        self.conv2a = GCNConv(int(hidden_channels), int(hidden_channels), cached=False, normalize=True)
        self.conv1b = GCNConv(int(hidden_channels), int(hidden_channels), cached=False, normalize=True)
        self.conv2b = GCNConv(int(hidden_channels), int(hidden_channels), cached=False, normalize=True)
        self.prelu1 = nn.PReLU(hidden_channels)
        self.prelu2 = nn.PReLU(hidden_channels)

        self.prelu1a = nn.PReLU(hidden_channels)
        self.prelu2a = nn.PReLU(hidden_channels)
        self.prelu1b = nn.PReLU(hidden_channels)
        self.prelu2b = nn.PReLU(hidden_channels)
        if type == 'dense':
            self.cat = CAT_DENSE(hidden_channels)
        else:
            self.cat = CAT_FIXED()

    def forward(self, x1, edge_index1, edge_weight1, x2, edge_index2, edge_weight2):
        x1 = self.conv1(x1, edge_index1, edge_weight1)
        x1 = self.prelu1(x1)
        x2 = self.conv2(x2, edge_index2, edge_weight2)
        x2 = self.prelu2(x2)
        if self.depth >= 2:
            x1 = self.conv1a(x1, edge_index1, edge_weight1)
            x1 = self.prelu1a(x1)

            x2 = self.conv2a(x2, edge_index2, edge_weight2)
            x2 = self.prelu2a(x2)
        if self.depth == 3:
            x1 = self.conv1b(x1, edge_index1, edge_weight1)
            x1 = self.prelu1b(x1)

            x2 = self.conv2b(x2, edge_index2, edge_weight2)
            x2 = self.prelu2b(x2)

        return self.cat(x1, x2)


class DGI_CAT(nn.Module):
    def __init__(self, n_in1, n_in2, n_h, type, depth, encoder):
        super(DGI_CAT, self).__init__()
        self.encoder = encoder(n_in1, n_in2, n_h, type,depth)
        self.activation = nn.PReLU()
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1a, edge_index1a, edge_weight1a, seq1b, edge_index1b, edge_weight1b, msk, samp_bias1,
                samp_bias2):
        h_1 = self.encoder(seq1a, edge_index1a, edge_weight1a, seq1b, edge_index1b, edge_weight1b)

        c = self.read(h_1, msk)
        c = self.sigm(c)
        seq2a, edge_index2a, edge_weight2a, seq2b, edge_index2b, edge_weight2b = corruption(seq1a, edge_index1a,
                                                                                            edge_weight1a, seq1b,
                                                                                            edge_index1b, edge_weight1b)
        h_2 = self.encoder(seq2a, edge_index2a, edge_weight2a, seq2b, edge_index2b, edge_weight2b)

        ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)

        return ret

    # Detach the return variables
    def embed(self, seq1a, edge_index1a, edge_weight1a, seq1b, edge_index1b, edge_weight1b, msk):
        h_1 = self.encoder(seq1a, edge_index1a, edge_weight1a, seq1b, edge_index1b, edge_weight1b)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()
