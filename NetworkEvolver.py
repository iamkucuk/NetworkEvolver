import os
import pickle
from collections import OrderedDict
from datetime import datetime

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader
from torchsummary import summary
from torchviz import make_dot

from ChromosomeCNN import ChromosomeCNN
from NetworkBuilder import ConvNet
from train_model import train_model


def evaluate(individual):
    torch.cuda.empty_cache()
    decoded_chromosome = individual.decode_chromosome()
    try:
        model = ConvNet(decoded_chromosome[1:])
        # torch.save(model, "hebe.pt")
        # dummy = torch.zeros([1, 3, 64, 64]).requires_grad_(True)
        # make_dot(model(dummy), params=dict(list(model.named_parameters()) + [('x', dummy)]))
        summary(model, input_size=(3, 64, 64), device="cpu")
    except ValueError as e:
        if str(e) == "Bad Network":
            return None, None

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

    data_dir = "data"

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              transformations[x])
                      for x in ['train', 'val', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32,
                                 shuffle=True)
                   for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer_name = decoded_chromosome[0]

    optimizer = None
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters())
    elif optimizer_name == "rmsprop":
        optimizer = RMSprop(model.parameters())

    criterion = nn.CrossEntropyLoss()

    now = datetime.now()
    model_name = now.strftime("%d%m%Y%H%M%S")

    # hl.build_graph(model, torch.zeros([1, 3, 64, 64]).to(device))

    return model_name, 1 / train_model(model_name, model, dataloaders, dataset_sizes, criterion, optimizer,
                                       num_epochs=10)


class NetworkEvolver:

    def __init__(self, mutation_rate=0.01, population_size=10, generations=10):
        self.generations = generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []
        self._initialize_population()
        self.best_fitness = 0

    def _initialize_population(self):
        for i in range(self.population_size):
            individual = ChromosomeCNN()
            individual.initialize_individual()
            self.population.append(individual)

    def repopulate(self, male, female):
        self.population = []
        for i in range(self.population_size):
            child_chromosome = male.mate(female)
            individual = ChromosomeCNN(child_chromosome)
            individual.mutate()
            self.population.append(individual)

    def breed(self):
        # find hall of fame
        fitnesses = {}

        for individual in self.population:
            name, fitness = evaluate(individual)
            if name is None:
                continue
            fitnesses.update({
                name: fitness,
            })

        # noinspection PyTypeChecker
        fitnesses = OrderedDict(sorted(fitnesses.items()), key=lambda x: x[1])

        male = fitnesses.popitem()
        female = fitnesses.popitem()

        torch.save(male, "male.pt")
        torch.save(female, "female.pt")

        if male[1] > self.best_fitness:
            self.best_fitness = male[1]
            torch.save(male, "best.pt")
        self.repopulate(male, female)

    def evolution(self):
        for i in range(self.generations):
            self.breed()


evolver = NetworkEvolver()
evolver.evolution()
