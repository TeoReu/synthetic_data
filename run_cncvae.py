import pandas as pd
import torch
import torch.nn.functional as F

from models.CNCVAE import CNCVAE
from sintetic_utils import extract_hetero_dataset, test_emb_quality, MMD

dataset1_train, dataset1_test, dataset2_train, dataset2_test, train_labels, test_labels = extract_hetero_dataset()


dataset = torch.cat([dataset1_train, dataset2_train], dim=1)

feature_dim = dataset.shape[1]
cncvae = CNCVAE(ds=128, ls=64, feature_dim=feature_dim)


optimizer = torch.optim.Adam(cncvae.parameters(), lr=0.001)


"""
Training the network for a given number of epochs
The loss after every epoch is printed
"""
beta = 10
for epoch in range(150):

    # Feeding a batch of images into the network to obtain the output image, mu, and logVar
    out, z, mu, logVar = cncvae(dataset1_train, dataset2_train)

    # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
    true_samples = torch.normal(0, 1, size=(dataset.size(0), 64))
    mmd = MMD(true_samples, z)
    loss = F.mse_loss(out, dataset)
    loss += mmd * beta

    # Backpropagation based on the loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch {}: Loss {}'.format(epoch, loss))



cncvae.eval()
z_train = cncvae.encode(dataset1_train, dataset2_train)
z_test = cncvae.encode(dataset1_test, dataset2_test)

train, test = test_emb_quality(z_train.detach(), train_labels, z_test.detach(), test_labels)

f = open('results.txt', 'a')
f.write('\ncncvae, ' + format(train[0]) + ', ' + format(test[0]) + ', ' + format(train[1]) + ', ' + format(
    test[1]) + ', ' + format(train[2]) + ', ' + format(test[2]))
f.close()