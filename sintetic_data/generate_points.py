import torch
import torch_geometric
from torch_geometric.utils import homophily
import torch.nn as nn
from models.cat_conv_encoder import Encoder
from models.infomax_weighted import DGI_CAT
import torch_geometric.transforms as T

from utils import plot_ax

d1_class1 = torch.normal(4, 30, size=(1000, 1000))
d1_class2 = torch.normal(2, 30, size=(1000, 1000))


d1_labels1 = torch.ones(1000)
d1_labels2 = torch.zeros(1000)
# test = [0] *400
# test = [0] *200 ...[1]*200
dataset1 = torch.cat([d1_class1, d1_class2], dim=0)
labels1 = torch.cat([d1_labels1, d1_labels2], dim=0)

d2_class1 = torch.normal(2.5, 30, size=(1000, 350))
d2_class2 = torch.normal(2.5, 30, size=(1000, 350))
dataset2 = torch.cat([d2_class1, d2_class2], dim=0)

e_1 = torch.randint(0, 1000, (2, 2000))
e_2 = torch.randint(1000, 2000, (2, 2000))
e_12 = torch.randint(0, 2000, (2, 2000))

edge_index_1 = torch.cat([e_1, e_2, e_12], dim=1)
edge_index_1 = torch.unique(edge_index_1, dim=1)

i_1 = torch.randint(0, 1000, (2, 2000))
i_2 = torch.randint(1000, 2000, (2, 2000))
i_12 = torch.randint(0, 2000, (2, 2000))

edge_index_2 = torch.cat([i_1, i_2, i_12], dim=1)
edge_index_2 = torch.unique(edge_index_2, dim=1)

print(homophily(edge_index_1, labels1))
print(homophily(edge_index_2, labels1))

# data = torch_geometric.data.Data(x=x, edge_index=edge_index, pos=x, dtype=torch.float)

data1 = torch_geometric.data.Data(x=dataset1, edge_index=edge_index_1, pos=dataset1, y=labels1, dtype=torch.float)
data2 = torch_geometric.data.Data(x=dataset2, edge_index=edge_index_2, pos=dataset2, y=labels1, dtype=torch.float)
gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=dict(method='ppr', alpha=0.25),
            sparsification_kwargs=dict(method='threshold',
                                       avg_degree=40), exact=True)
dataset1 = gdc(data1)
#test_dataset_1 = gdc(test_data_1)
edge_weight1 = data1.edge_attr

dataset2 = gdc(data2)
#test_dataset_2 = gdc(test_data_2)
edge_weight2 = data2.edge_attr
model = DGI_CAT(data1.num_features, data2.num_features, 64, encoder=Encoder)

b_xent = nn.BCEWithLogitsLoss()
lbl_1 = torch.ones(data1.num_nodes)
lbl_2 = torch.zeros(data1.num_nodes)
lbl = torch.cat((lbl_1, lbl_2), 0)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

plot_ax(torch.cat([dataset1.x,dataset2.x], dim=1).numpy(), data1.y, 0, 'label', 0, 'sintetic_data', 0, 0)

def train(epoch):
    model.train()
    optimizer.zero_grad()
    logits = model(data1.x, data1.edge_index, edge_weight1, data2.x, data2.edge_index, edge_weight2, None, None, None)
    # param = model.named_parameters()
    pos_z = model.embed(data1.x, data1.edge_index, edge_weight1, data2.x, data2.edge_index, edge_weight2, None)[0]
    if epoch % 20 == 1:
        plot_ax(pos_z.detach().numpy(), data1.y, epoch, 'label', 0, 'sintetic_data', epoch,
                0)
    loss = b_xent(logits, lbl)
    loss.backward()
    optimizer.step()
    # plot_grad_flow(model.named_parameters())
    return loss.item()

best = 1e9
patience = 200
best_t = 0
for epoch in range(1, 1000):
    loss = train(epoch)
    # print(gnn_model_summary(model))
    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        #torch.save(model.state_dict(), 'best_dgi.pkl')
    else:
        cnt_wait += 1
    if cnt_wait == patience:
        print('Early stopping!')
        break

    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')