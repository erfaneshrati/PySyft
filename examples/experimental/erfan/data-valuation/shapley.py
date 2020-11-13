import numpy as np
import torch
from mnist import MNIST


class Shapley:
    def __init__(self, n_test=50):
        mnist = MNIST()

        use_cuda = torch.cuda.is_available()
        torch.manual_seed(0)
        device = torch.device("cuda" if use_cuda else "cpu")

        self.mnist_embeds = torch.tensor(mnist.test_embeds[0:500, 0:128], device=device)
        self.mnist_labels = torch.tensor(mnist.test_embeds[0:500, 128], device=device)
        self.train_x = self.mnist_embeds[n_test:]
        self.test_x = self.mnist_embeds[0:n_test]
        self.train_y = self.mnist_labels[n_test:]
        self.test_y = self.mnist_labels[0:n_test]

    def unencrypted_sv_unweighted_knn(self, K=5):
        n_train = len(self.train_x)
        n_test = len(self.test_x)
        s = np.zeros((n_test, n_train))
        for j, e_test in enumerate(self.test_x):
            dists = torch.sum((self.train_x - e_test)**2, dim=1)
            _, inds = torch.sort(dists, descending=True)
            inds = inds.cpu().numpy()
            for i, ind in enumerate(inds):
                if i == 0:
                    s[j, inds[0]] = 1/n_train if (self.train_y[inds[0]]==self.test_y[j]) else 0
                else:
                    s[j, ind] = s[j, inds[i-1]] + \
                                ((float(self.train_y[ind]==self.test_y[j]) - \
                                  float(self.train_y[inds[i-1]]==self.test_y[j])) * \
                                  min(K,(n_train-i))/(K*(n_train-i)))
        s = np.mean(s, axis=0)
        print (s)
Shapley().unencrypted_sv_unweighted_knn()
