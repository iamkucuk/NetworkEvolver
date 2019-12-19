import os
import random
import time
from datetime import datetime

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn, optim
from torch.optim.rmsprop import RMSprop
from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad

from torch.utils.data import DataLoader
from ChromosomeCNN import ChromosomeCNN
from NetworkBuilder import ConvNet
from train_model import train_model


def evaluate(individual):
    decoded_chromosome = individual.decode_choromosome()
    model = ConvNet(decoded_chromosome[1:])

    transformations = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    data_dir = None

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              transformations[x])
                      for x in ['train', 'val', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=100,
                                 shuffle=True)
                   for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = model.to(device)

    optimizer_name = decoded_chromosome[0]

    optimizers = {
        'adam': optim.Adam(model.parameters()),
        'rmsprop': RMSprop(model.parameters()),
        'adagrad': Adagrad(model.parameters()),
        'adadelta': Adadelta(model.parameters()),
    }
    optimizer = optimizers[optimizer_name]
    criterion = nn.CrossEntropyLoss()

    return 1/train_model(str(datetime.now().time()), model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=10)


class NetworkEvolver:

    def __init__(self, mutation_rate=0.01, population_size=10, generations=10):
        self.generations = generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []
        self._initialize_population()

    def _initialize_population(self):
        for i in range(self.population_size):
            individual = ChromosomeCNN()
            individual.initialize_individual()
            self.population.append(individual)

    def repopulate(self, male, female):
        self.population = []
        for i in range(self.population_size):
            child_chromosome = male.mate(female)
            individual = ChromosomeCNN()
            individual.chromosome = child_chromosome
            individual.mutate()
            self.population.append(individual)

    def breed(self):
        # find hall of fame
        fitnesses = []

        for individual in self.population:
            fitnesses.append(evaluate(individual))

        # self.repopulate(male=male, female=female)
