from sintetic_utils import extract_dataset, pca, test_emb_quality

train_dataset, train_labels, test_dataset, test_labels = extract_dataset()
train_emb = pca(train_dataset)
test_emb = pca(test_dataset)


train, test = test_emb_quality(train_emb, train_labels, test_emb, test_labels)
f = open('results.txt', 'a')
f.write('\npca, ' + format(train[0]) + ', ' + format(test[0]) + ', ' + format(train[1]) + ', ' + format(
    test[1]) + ', ' + format(train[2]) + ', ' + format(test[2]))
f.close()