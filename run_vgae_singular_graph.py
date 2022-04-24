import argparse

import pandas as pd
import torch
from torch_geometric.nn import VGAE, GCNConv
from torch_geometric.utils import homophily

from sintetic_utils import build_graph, extract_dataset, test_emb_quality, MMD


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super().__init__()
        self.depth = depth
        self.conv1 = GCNConv(in_channels, 2 * out_channels, normalize=True)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, normalize=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, normalize=True)
        self.conv2 = GCNConv(2 * out_channels, 2 * out_channels)
        self.conv3 = GCNConv(2 * out_channels, 2 * out_channels)

    def forward(self, X, edge_index, edge_weight):
        x = self.conv1(X, edge_index, edge_weight).relu()
        if self.depth >= 2:
            x = self.conv2(x, edge_index, edge_weight).relu()
        if self.depth == 3:
            x = self.conv3(x, edge_index, edge_weight).relu()
        return self.conv_mu(x, edge_index, edge_weight), self.conv_logstd(x, edge_index, edge_weight)


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

ls = 128
model = VGAE(encoder=VariationalGCNEncoder(train_graph.num_features, ls, args.depth))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = train_graph
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# print(model.named_parameters())

# print(summary(model, [(1000, 350), (2, 1, 1000), (1,1000), (1000, 1350), (2, 1, 1200), (1,1200)], [torch.float, torch.long, torch.float, torch.float, torch.long, torch.float]))
# print(gnn_model_summary(model))
beta = 1


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_graph.x, train_graph.edge_index, train_graph.edge_attr)
    loss = model.recon_loss(z, train_graph.edge_index)
    if args.loss == 'kl':
        loss = loss + (1 / train_graph.num_nodes) * model.kl_loss()
    else:
        true_samples = torch.normal(0, 1, size=(train_graph.num_nodes, ls))
        loss = loss + beta * MMD(true_samples, z)
    loss.backward()
    optimizer.step()
    return float(loss)


for epoch in range(1, 150):
    loss = train()
    # auc, ap = test(test_data)
    print(f'Epoch: {epoch:03d}, LOSS: {loss:.4f}')

# tmodel.load_state_dict(torch.load('dgi_singular_graph.pkl'))

z_train = model.encode(train_graph.x, train_graph.edge_index, train_graph.edge_attr)
z_test = model.encode(test_graph.x, test_graph.edge_index, test_graph.edge_attr)

train, test = test_emb_quality(z_train.detach(), train_labels, z_test.detach(), test_labels)

f = open('results.txt', 'a')
f.write('\nvgae_singular_graph' + ',' + format(args.depth) + ',' + format(args.loss) + ',' + format(
    train[0]) + ', ' + format(test[0]) + ', ' + format(
    train[1]) + ', ' + format(test[1]) + ', ' + format(train[2]) + ', ' + format(test[2]) + ',' + format(
    homophily(train_graph.edge_index, train_labels)))
f.close()
