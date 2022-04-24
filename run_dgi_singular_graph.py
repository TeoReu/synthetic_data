import argparse

import pandas as pd
import torch
from torch_geometric.nn import VGAE, DeepGraphInfomax, GCNConv
from torch_geometric.utils import homophily

from sintetic_utils import build_graph, extract_dataset, test_emb_quality


def corruption(x, edge_index, edge_weight):
    return x[torch.randperm(x.size(0))], edge_index, edge_weight


def unweighted_corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, depth):
        super().__init__()
        self.depth = depth
        self.conv = GCNConv(in_channels, hidden_channels, cached=False, normalize=True)
        self.prelu = torch.nn.PReLU(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, cached=False, normalize=True)
        self.prelu2 = torch.nn.PReLU(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels, cached=False, normalize=True)
        self.prelu3 = torch.nn.PReLU(hidden_channels)
    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        x = self.prelu(x)
        if self.depth >= 2:
            x = self.conv2(x, edge_index, edge_weight)
            x = self.prelu2(x)
        if self.depth == 3:
            x = self.conv3(x, edge_index, edge_weight)
            x = self.prelu3(x)
        return x



parser = argparse.ArgumentParser()
parser.add_argument('--h1', type=float, default=1)
parser.add_argument('--h2', type=float)
parser.add_argument('--depth', type=int)
parser.add_argument('--loss', type=str, default='kl')

if __name__ == "__main__":
    args = parser.parse_args()

train_dataset, train_labels, test_dataset, test_labels = extract_dataset()
train_graph = build_graph(train_dataset, train_labels, args.h2)
test_graph = build_graph(test_dataset, test_labels, args.h2)

model = DeepGraphInfomax(
    hidden_channels=64, encoder=Encoder(train_graph.num_features, 64, args.depth),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

torch.manual_seed(47)

cnt_wait = 0
patience = 100
best = 1e9
best_t = 0


def train(epoch):
    model.train()
    optimizer.zero_grad()
    pos_z, neg_z, summary = model(train_graph.x, train_graph.edge_index, train_graph.edge_attr)
    loss = model.loss(pos_z, neg_z, summary)
    # if epoch % 50 == 0:
    #    plot_ax(pos_z.detach().numpy(), train_data.y, epoch, args.graph_construction, hp)
    loss.backward()
    optimizer.step()
    return loss.item()


for epoch in range(1, 150):
    loss = train(epoch)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), 'dgi_singular_graph.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

model.load_state_dict(torch.load('dgi_singular_graph.pkl'))

model.eval()
z_train, _, _ = model(train_graph.x, train_graph.edge_index, train_graph.edge_attr)
z_test, _, _ = model(test_graph.x, test_graph.edge_index, test_graph.edge_attr)

train, test = test_emb_quality(z_train.detach(), train_labels, z_test.detach(), test_labels)

f = open('results.txt', 'a')
f.write('\ndgi_singular_graph' + ',' + format(args.depth) + ',' + format(
    train[0]) + ', ' + format(test[0]) + ', ' + format(
    train[1]) + ', ' + format(test[1]) + ', ' + format(train[2]) + ', ' + format(test[2]) + ',' + format(
    homophily(train_graph.edge_index, train_labels)))
f.close()
