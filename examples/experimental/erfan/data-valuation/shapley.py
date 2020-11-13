import numpy as np
from mnist import MNIST


class Shapley:
    def __init__(self, ):
        mnist = MNIST()
        self.mnist_embeds = mnist.test_embeds[:,0:128]
        self.mnist_labels = mnist.test_embeds[:,128]

Shapley()
