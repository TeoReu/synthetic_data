import torch
from torch import nn
import torch.nn.functional as F


class CNCVAE(nn.Module):
    def __init__(self, feature_dim, ds, ls):
        super(CNCVAE, self).__init__()
        self.enc = nn.Linear(feature_dim, ds)
        self.mean = nn.Linear(ds, ls)
        self.std = nn.Linear(ds, ls)
        self.dec1 = nn.Linear(ls, ds)
        self.dec2 = nn.Linear(ds, feature_dim)

    def encoder(self, x1, x2):
        x = F.relu(torch.concat([x1, x2], dim=-1))
        x = F.relu(self.enc(x))
        z_mean = self.mean(x)
        z_std = self.std(x)

        return z_mean, z_std

    def reparametrize(self, mean, std):
        s = torch.exp(std / 2)
        eps = torch.randn_like(s)
        return mean + std * eps

    def decoder(self, z):
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))

        return x

    def forward(self, x1, x2):
        mean, std = self.encoder(x1, x2)
        z = self.reparametrize(mean, std)
        out = self.decoder(z)
        return out, z, mean, std

    def encode(self, x1, x2):
        mean, std = self.encoder(x1, x2)

        return self.reparametrize(mean, std)
