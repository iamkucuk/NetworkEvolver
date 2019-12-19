import numpy as np
import random
import math

class Genome:
    """
    Awesome class definition
    """

    def __init__(self, max_conv_layers, max_dense_layers, max_filters,
                 max_dense_nodes, input_shape, n_classes,
                 batch_normalization=True, dropout=True, max_pooling=True,
                 optimizers=None, activations=None):
        """
            Awesome parameter explainations
        """

        if max_filters > 0:
            filter_range_max = int(math.log(max_filters, 2)) + 1

        self.optimizer = [
            'adam',
            'rmsprop',
            'adagrad',
            'adadelta'
        ]
        self.activation = [
            'relu',
            'sigmoid',
        ]
        self.convolutional_layer_shape = [
            "active",
            "num filters",
            "batch normalization",
            "activation",
            "dropout",
            "max pooling",
        ]
        self.dense_layer_shape = [
            "active",
            "num nodes",
            "batch normalization",
            "activation",
            "dropout",
        ]
        self.layer_params = {
            "active": [0, 1],
            "num filters": [2 ** i for i in range(3, filter_range_max)],
            "num nodes": [2 ** i for i in range(4, int(math.log(max_dense_nodes, 2)) + 1)],
            "batch normalization": [0, (1 if batch_normalization else 0)],
            "activation": list(range(len(self.activation))),
            "dropout": [(i if dropout else 0) for i in range(11)],
            "max pooling": list(range(3)) if max_pooling else 0,
        }

        self.convolution_layers = max_conv_layers
        self.convolution_layer_size = len(self.convolutional_layer_shape)
        self.dense_layers = max_dense_layers - 1 # this doesn't include the softmax layer, so -1
        self.dense_layer_size = len(self.dense_layer_shape)
        self.input_shape = input_shape
        self.n_classes = n_classes

    def decode(self, genome):

        #Check for if genome is compatible

        
    