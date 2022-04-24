import networkx as nx
import pandas as pd
import torch
import torch_geometric
import torch_geometric.transforms as T
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from torch_geometric.data import HeteroData


def build_graph(dataset, labels, h=1, inter=1):
    nr_samples = labels.size()[0]
    nr_class1 = labels[labels == 1].size()[0]
    e_1 = torch.randint(0, nr_class1 - 1, (2, int(nr_samples*2*h)))
    e_2 = torch.randint(nr_class1, nr_samples, (2, int(nr_samples*2*h)))
    e_12 = torch.randint(0, nr_samples, (2, int(nr_samples*2*inter)))

    edge_index = torch.cat([e_1, e_2, e_12], dim=1)
    edge_index = torch.unique(edge_index, dim=1)

    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',

                diffusion_kwargs=dict(method='ppr', alpha=0.25),
                sparsification_kwargs=dict(method='threshold',
                                           avg_degree=40), exact=True)
    graph = torch_geometric.data.Data(x=dataset, edge_index=edge_index, pos=dataset, y=labels, dtype=torch.float)

    return gdc(graph)


def build_hetero_graph(graph1, graph2):
    data = HeteroData()

    data['1'].x = torch.cat([graph1.x, torch.zeros(graph2.num_nodes, graph2.num_features)], dim=1)
    data['2'].x = torch.cat([torch.zeros(graph1.num_nodes, graph1.num_features), graph2.x], dim=1)

    data['1', 'close', '1'].edge_index = graph1.edge_index
    data['1', 'close', '1'].edge_attr = graph1.edge_attr

    data['2', 'close', '2'].edge_index = graph2.edge_index
    data['2', 'close', '2'].edge_attr = graph2.edge_attr


    node_to_node = torch.zeros((2, graph1.num_nodes), dtype=torch.int)
    for i in range(graph1.num_nodes):
        node_to_node[0][i] = int(i)
        node_to_node[1][i] = int(i)

    data['1', 'is', '2'].edge_index = node_to_node
    data['1','is','2'].edge_attr = torch.zeros(graph1.num_nodes)
    data = T.ToUndirected()(data)
    #data = T.AddSelfLoops()(data)

    data = data.to_homogeneous()

    return data


def pca(x):
    tsneRAW = TSNE(random_state=42, perplexity=40)
    x = tsneRAW.fit_transform(x)
    return x

def sintetic_plot_ax(x, y, title, e):
    tsneRAW = TSNE(random_state=42, perplexity=40)
    x = tsneRAW.fit_transform(x)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.scatter(x[:, 0], x[:, 1], c=y, s=4)
    fig.savefig(
        'plots/' + format(title) + 'epoch_' + format(
            e) + '.jpg', dpi=400)


def extract_dataset():
    dataset1_train = pd.read_csv('/Users/teodorareu/PycharmProjects/pythonProject3/sintetic_data'
                                 '/dataset_1_train.cvs', sep=" ", index_col=None)
    dataset1_train = torch.tensor(dataset1_train.values, dtype=torch.float32)

    dataset2_train = pd.read_csv('/Users/teodorareu/PycharmProjects/pythonProject3/sintetic_data'
                                 '/dataset_2_train.cvs', sep=" ", index_col=None)
    dataset2_train = torch.tensor(dataset2_train.values, dtype=torch.float32)

    train_dataset = torch.cat([dataset1_train, dataset2_train], dim=1)

    train_labels = pd.read_csv('/Users/teodorareu/PycharmProjects/pythonProject3/sintetic_data'
                               '/train_labels.cvs', sep=" ", index_col=None)

    train_labels = torch.tensor(train_labels.values, dtype=torch.float32)

    dataset1_test = pd.read_csv('/Users/teodorareu/PycharmProjects/pythonProject3/sintetic_data'
                                '/dataset_1_test.cvs', sep=" ", index_col=None)
    dataset1_test = torch.tensor(dataset1_test.values, dtype=torch.float32)

    dataset2_test = pd.read_csv('/Users/teodorareu/PycharmProjects/pythonProject3/sintetic_data'
                                '/dataset_2_test.cvs', sep=" ", index_col=None)
    dataset2_test = torch.tensor(dataset2_test.values, dtype=torch.float32)

    test_dataset = torch.cat([dataset1_test, dataset2_test], dim=1)

    test_labels = pd.read_csv('/Users/teodorareu/PycharmProjects/pythonProject3/sintetic_data'
                              '/test_labels.cvs', sep=" ", index_col=None)

    test_labels = torch.tensor(test_labels.values, dtype=torch.float32)

    return train_dataset, train_labels, test_dataset, test_labels

def extract_hetero_dataset():
    dataset1_train = pd.read_csv('/Users/teodorareu/PycharmProjects/pythonProject3/sintetic_data'
                                 '/dataset_1_train.cvs', sep=" ", index_col=None)
    dataset1_train = torch.tensor(dataset1_train.values, dtype=torch.float32)

    dataset2_train = pd.read_csv('/Users/teodorareu/PycharmProjects/pythonProject3/sintetic_data'
                                 '/dataset_2_train.cvs', sep=" ", index_col=None)
    dataset2_train = torch.tensor(dataset2_train.values, dtype=torch.float32)


    train_labels = pd.read_csv('/Users/teodorareu/PycharmProjects/pythonProject3/sintetic_data'
                               '/train_labels.cvs', sep=" ", index_col=None)

    train_labels = torch.tensor(train_labels.values, dtype=torch.float32)

    dataset1_test = pd.read_csv('/Users/teodorareu/PycharmProjects/pythonProject3/sintetic_data'
                                '/dataset_1_test.cvs', sep=" ", index_col=None)
    dataset1_test = torch.tensor(dataset1_test.values, dtype=torch.float32)

    dataset2_test = pd.read_csv('/Users/teodorareu/PycharmProjects/pythonProject3/sintetic_data'
                                '/dataset_2_test.cvs', sep=" ", index_col=None)
    dataset2_test = torch.tensor(dataset2_test.values, dtype=torch.float32)


    test_labels = pd.read_csv('/Users/teodorareu/PycharmProjects/pythonProject3/sintetic_data'
                              '/test_labels.cvs', sep=" ", index_col=None)

    test_labels = torch.tensor(test_labels.values, dtype=torch.float32)
    return dataset1_train, dataset1_test, dataset2_train, dataset2_test, train_labels, test_labels


def test_emb_quality(emb_train, train_labels, emb_test, test_labels):
    accsTrain = []
    accsTest = []
    nb = GaussianNB()
    nb.fit(emb_train, train_labels)

    x_p_classes = nb.predict(emb_train)
    accTrain = accuracy_score(train_labels, x_p_classes)
    accsTrain.append(accTrain)

    y_p_classes = nb.predict(emb_test)
    accsTest.append(accuracy_score(test_labels, y_p_classes))

    svm = SVC(C=1.5, kernel='rbf', random_state=20, gamma='auto')
    svm.fit(emb_train, train_labels)

    x_p_classes = svm.predict(emb_train)
    accTrain = accuracy_score(train_labels, x_p_classes)
    accsTrain.append(accTrain)

    y_p_classes = svm.predict(emb_test)
    accsTest.append(accuracy_score(test_labels, y_p_classes))

    rf = RandomForestClassifier(n_estimators=50, random_state=42,
                                max_features=.5)
    rf.fit(emb_train, train_labels)

    x_p_classes = rf.predict(emb_train)
    accTrain = accuracy_score(train_labels, x_p_classes)
    accsTrain.append(accTrain)

    y_p_classes = rf.predict(emb_test)
    accsTest.append(accuracy_score(test_labels, y_p_classes))

    return accsTrain, accsTest

def compute_kernel(x, y):
    x_size = torch.tensor(x.shape[0])
    y_size = torch.tensor(y.shape[0])
    t1 = torch.tensor(1)
    dim = torch.tensor(x.shape[1])
    x_stacked_1 = list(torch.stack([x_size, t1, t1]))
    y_stacked_1 = list(torch.stack([t1, y_size, t1]))
    x_stacked_d = list(torch.stack([x_size, t1, dim]))
    y_stacked_d = list(torch.stack([t1, y_size, dim]))
    tiled_x = torch.tile(torch.reshape(x, x_stacked_d), y_stacked_1)
    tiled_y = torch.tile(torch.reshape(y, y_stacked_d), x_stacked_1)

    kernel_input = torch.exp(-torch.mean((tiled_x - tiled_y)**2, dim=2)) / dim

    return  kernel_input


def MMD(x, y):
    XX = compute_kernel(x, x)
    YY = compute_kernel(y, y)
    XY = compute_kernel(x, y)

    return torch.mean(XX) + torch.mean(YY) - 2. * torch.mean(XY)

def draw_graph(data):
    g = torch_geometric.utils.to_networkx(data)
    # g.add_edges_from(data.edge_index)
    # pretty colours: #db9cc3, #d17c62
    g.remove_edges_from(nx.selfloop_edges(g))

    nx.draw_spring(g.to_undirected(reciprocal=False, as_view=False), node_size=10,
                   node_color=data.y)
    plt.show()