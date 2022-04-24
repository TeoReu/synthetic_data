import argparse

import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import DeepGraphInfomax, GCNConv
from torch_geometric.utils import homophily

from sintetic_utils import build_graph, test_emb_quality, extract_hetero_dataset

def corruption(x, edge_index, edge_weight):
    return x[torch.randperm(x.size(0))], edge_index, edge_weight


def unweighted_corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels, cached=False, normalize=True)
        self.prelu = torch.nn.PReLU(hidden_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv(x, edge_index, edge_weight)
        x = self.prelu(x)
        return x

parser = argparse.ArgumentParser()
parser.add_argument('--h1', type=float, default= 1)
parser.add_argument('--h2', type=float)
if __name__ == "__main__":
    args = parser.parse_args()

dataset1_train, dataset1_test, dataset2_train, dataset2_test, train_labels, test_labels = extract_hetero_dataset()
train_graph1 = build_graph(dataset1_train, train_labels, args.h1)
train_graph2 = build_graph(dataset2_train, train_labels, args.h2)

test_graph1 = build_graph(dataset1_test, test_labels, args.h1)
test_graph2 = build_graph(dataset2_test, test_labels, args.h2)


model = DeepGraphInfomax(
    hidden_channels=64, encoder=Encoder(train_graph1.num_features, 64),
    summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
    corruption=corruption)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# plot_ax(torch.cat([dataset1.x,dataset2.x], dim=1).numpy(), data1.y, 0, 'label', 0, 'sintetic_data', 0, 0)
for train_graph, test_graph, i in zip([train_graph1, train_graph2], [test_graph1, test_graph2], [1,2]):
    model = DeepGraphInfomax(
        hidden_channels=64, encoder=Encoder(train_graph.num_features, 64),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def train(epoch):
        model.train()
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(train_graph.x, train_graph.edge_index, train_graph.edge_attr)
        loss = model.loss(neg_z, pos_z, summary)
        loss.backward()
        optimizer.step()
        # plot_grad_flow(model.named_parameters())
        return loss.item()


    best = 1e9
    patience = 200
    best_t = 0
    for epoch in range(1, 150):
        loss = train(epoch)
        # print(gnn_model_summary(model))
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'dgi_1_graph_'+format(i)+'.pkl')
        else:
            cnt_wait += 1
        if cnt_wait == patience:
            print('Early stopping!')
            break

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    model.load_state_dict(torch.load('dgi_1_graph_'+format(i)+'.pkl'))

    model.eval()
    z_train, _, _ = model(train_graph.x, train_graph.edge_index, train_graph.edge_attr)
    z_test, _, _ = model(test_graph.x, test_graph.edge_index, test_graph.edge_attr)

    train, test = test_emb_quality(z_train.detach(), train_labels, z_test.detach(), test_labels)

    f = open('results.txt', 'a')
    f.write(
        '\ndgi_1_graph_'+format(i)+', '+ format(train[0]) + ', ' + format(test[0]) + ', ' + format(train[1]) + ', ' + format(
            test[1]) + ', ' + format(train[2]) + ', ' + format(test[2]) + ',' + format(
    homophily(train_graph1.edge_index, train_labels)) + ',' + format(homophily(train_graph2.edge_index, train_labels)))
    f.close()
