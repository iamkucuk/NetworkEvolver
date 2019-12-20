import numpy as np
import random
import math


class ChromosomeCNN:
    """
    Awesome class definition
    """

    def __init__(self, chromosome=None, max_conv_layers=26, max_filters=1024,
                 input_shape=64, n_classes=200,
                 optimizers=None, activations=None):
        """
            Awesome parameter explainations
        """

        self.chromosome = []
        if chromosome is not None:
            self.chromosome = chromosome
        self.genome_length = 5

        self.chromosome_length = max_conv_layers * self.genome_length + 3

        self.input_shape = input_shape
        self.n_classes = n_classes
        self.max_conv_layers = max_conv_layers
        self.max_filters = int(math.log2(max_filters))

        self.optimizer = [
            'adam',
            'rmsprop',
            'adagrad',
            'adadelta',
        ]

        self.activation = [
            'relu',
            'sigmoid',
            'tanh',
        ]

        self.layer_names = [
            "dense",
            "max",
            "avg",
            "concat",
            "sum",
            "conv",
            "res",
        ]

        self.layer_params = {
            # "input1": 0,
            # "input2": 0,
            "active": [0, 1],
            "kernel_size": [3, 5, 7],
            "layer_type": list(range(7)),
            "num_filters": [2 ** i for i in range(3, self.max_filters)],
            "batch_normalization": [0, 1],
            "activation": list(range(len(self.activation))),
            "padding": [0, 1],
        }

    def decode_chromosome(self):
        # Check for if genome is compatible

        decoded_chromosome = [self.optimizer[self.chromosome[0]], [
            ["conv", self.layer_params["num_filters"][self.chromosome[1]],
             self.layer_params["kernel_size"][self.chromosome[2]]],
            0, 0
        ]]

        for i in range(self.max_conv_layers):
            genome = self.chromosome[i * self.genome_length + 3: (i + 1) * self.genome_length + 3]
            # check if connected
            layer_params = [self.layer_names[genome[0]], self.layer_params["num_filters"][genome[1]],
                            self.layer_params["kernel_size"][genome[2]], 0, 0]

            decoded_genome = [layer_params, genome[-2], genome[-1]]

            decoded_chromosome.append(decoded_genome)

        return decoded_chromosome

    def mutate(self, mutation_rate=.01, number_of_max_mutation=3):
        for i in range(number_of_max_mutation):
            if random.random() < mutation_rate:
                selection = round(random.uniform(0, self.chromosome_length))
                if selection == 0:
                    new_gene = round(random.uniform(0, 3))
                elif selection == 1 or ((selection - 3) % 5 == 1):
                    new_gene = round(random.uniform(0, self.max_filters - 3))
                elif selection == 2 or ((selection - 3) % 5 == 2):
                    new_gene = round(random.uniform(0, 2))
                elif (selection - 3) % 5 == 0:
                    new_gene = round(random.uniform(0, 6))
                else:
                    new_gene = round(random.uniform(0, selection))

                self.chromosome[selection] = new_gene

    def initialize_individual(self):
        self.chromosome = []
        for selection in range(self.chromosome_length):
            if selection == 0:
                new_gene = round(random.uniform(0, 3))
            elif selection == 1 or ((selection - 3) % 5 == 1):
                new_gene = round(random.uniform(0, self.max_filters - 3))
            elif selection == 2 or ((selection - 3) % 5 == 2):
                new_gene = round(random.uniform(0, 2))
            elif (selection - 3) % 5 == 0:
                new_gene = round(random.uniform(0, 6))
            else:
                new_gene = round(random.uniform(0, selection))

            self.chromosome.append(new_gene)

    def mate(self, partner):
        cut_point = round(random.uniform(1, self.chromosome_length - 1))
        return self.chromosome[:cut_point].extend(partner.chromosome[cut_point:])

    def _check_if_active(self, layer):
        pass
