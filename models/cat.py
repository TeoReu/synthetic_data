import torch
from torch import nn


class CAT3_DENSE(nn.Module):
    def __init__(self, n_h):
        super(CAT3_DENSE, self).__init__()
        self.dense = nn.Sequential(nn.Linear(3 * n_h, n_h), nn.ReLU())

    def forward(self, h_1, h_2, h_3):
        output = torch.cat((h_1, h_2, h_3), dim=1)
        output = self.dense(output)
        return output


class CAT_DENSE(nn.Module):
    def __init__(self, n_h):
        super(CAT_DENSE, self).__init__()
        self.dense = nn.Sequential(nn.Linear(2 * n_h, n_h), nn.ReLU())

    def forward(self, h_1, h_2):
        output = torch.cat((h_1, h_2), dim=1)
        output = self.dense(output)
        return output


class CAT_FIXED(nn.Module):
    def __init__(self):
        super(CAT_FIXED, self).__init__()

    def forward(self, h_1, h_2):
        output = (h_1 + h_2) / 2
        return output
