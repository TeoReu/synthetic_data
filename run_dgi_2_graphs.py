import argparse

import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.utils import homophily

from models.TwoGraphsDGI import Encoder, DGI_CAT
from sintetic_utils import build_graph, test_emb_quality, extract_hetero_dataset, draw_graph

parser = argparse.ArgumentParser()
parser.add_argument('--cat')
parser.add_argument('--h1', type=float, default= 1)
parser.add_argument('--h2', type=float)
parser.add_argument('--depth', type=int)
if __name__ == "__main__":
    args = parser.parse_args()

dataset1_train, dataset1_test, dataset2_train, dataset2_test, train_labels, test_labels = extract_hetero_dataset()
train_graph1 = build_graph(dataset1_train, train_labels, args.h1, 0.1)
train_graph2 = build_graph(dataset2_train, train_labels, args.h2, 0.1)

draw_graph(train_graph1)
draw_graph(train_graph2)

test_graph1 = build_graph(dataset1_test, test_labels, args.h1)
test_graph2 = build_graph(dataset2_test, test_labels, args.h2)

#print(homophily(train_graph1.edge_index, train_labels))
#print(homophily(train_graph2.edge_index, train_labels))
#print(homophily(test_graph1.edge_index, test_labels))
#print(homophily(test_graph2.edge_index, test_labels))

b_xent = nn.BCEWithLogitsLoss()
lbl_1 = torch.ones(train_graph1.num_nodes)
lbl_2 = torch.zeros(train_graph1.num_nodes)
lbl = torch.cat((lbl_1, lbl_2), 0)

model = DGI_CAT(train_graph1.num_features, train_graph2.num_features, 64, args.cat, args.depth, encoder=Encoder)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# plot_ax(torch.cat([dataset1.x,dataset2.x], dim=1).numpy(), data1.y, 0, 'label', 0, 'sintetic_data', 0, 0)

def train(epoch):
    model.train()
    optimizer.zero_grad()
    logits = model(train_graph1.x, train_graph1.edge_index, train_graph1.edge_attr, train_graph2.x,
                   train_graph2.edge_index, train_graph2.edge_attr, None, None, None)
    # param = model.named_parameters()
    # pos_z = model.embed(data1.x, data1.edge_index, edge_weight1, data2.x, data2.edge_index, edge_weight2, None)[0]
    # if epoch % 20 == 1:
    #    plot_ax(pos_z.detach().numpy(), data1.y, epoch, 'label', 0, 'sintetic_data', epoch,
    #            0)
    loss = b_xent(logits, lbl)
    loss.backward()
    optimizer.step()
    # plot_grad_flow(model.named_parameters())
    return loss.item()


best = 1e9
patience = 50
best_t = 0
for epoch in range(1, 150):
    loss = train(epoch)
    # print(gnn_model_summary(model))
    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), 'dgi_2_graphs.pkl')
    else:
        cnt_wait += 1
    if cnt_wait == patience:
        print('Early stopping!')
        break

    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

model.load_state_dict(torch.load('dgi_2_graphs.pkl'))

model.eval()
z_train = model.embed(train_graph1.x, train_graph1.edge_index, train_graph1.edge_attr, train_graph2.x,
                      train_graph2.edge_index, train_graph2.edge_attr, None)[0]
z_test = model.embed(test_graph1.x, test_graph1.edge_index, test_graph1.edge_attr, test_graph2.x,
                     test_graph2.edge_index, test_graph2.edge_attr, None)[0]

train, test = test_emb_quality(z_train.detach(), train_labels, z_test.detach(), test_labels)

f = open('results.txt', 'a')
f.write('\ndgi_2_graph' + ',' + format(args.cat) + ',' + format(args.depth) + ',' + format(train[0]) + ', ' + format(test[0]) + ', ' + format(
    train[1]) + ', ' + format(test[1]) + ', ' + format(train[2]) + ', ' + format(test[2]) + ',' + format(
    homophily(train_graph1.edge_index, train_labels)) + ',' + format(homophily(train_graph2.edge_index, train_labels)))
f.close()
