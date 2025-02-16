from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import torch
import pickle


def get_accuracy(model, train_loader, test_loader):
    X = []
    Y = []

    for batch in train_loader:
        x, y = model.transform_batch(batch)
        with torch.inference_mode():
            embeddings = model(x)

        X.append(embeddings.cpu())
        Y.append(y.cpu())

    Xs = torch.cat(X)
    Ys = torch.cat(Y)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(Xs, Ys)

    X_test = []
    Y_test = []

    for batch in test_loader:
        x, y = model.transform_batch(batch)
        with torch.inference_mode():
            embeddings = model(x)

        X_test.append(embeddings.cpu())
        Y_test.append(y.cpu())

    Xs_test = torch.cat(X_test)
    Ys_test = torch.cat(Y_test)

    predictions = knn.predict(Xs_test)

    return accuracy_score(predictions, Ys_test)


def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.

    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                pass
                # raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                #                    'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.shape))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))
