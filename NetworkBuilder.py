import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding="same", activation="relu",
                 batch_norm=True):
        super(ConvBlock, self).__init__()

        self.batch_norm = batch_norm
        padding_size = 0

        if padding == "same":
            padding_size = kernel_size // 2

        self.layer = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding=padding_size,
                               stride=stride)

        if batch_norm:
            self.batch_norm_layer = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = self.layer(inputs)
        if self.batch_norm:
            x = self.batch_norm_layer(x)

        return self.relu(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=True):
        super(ResBlock, self).__init__()

        padding_size = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding=padding_size,
                               stride=stride)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding=padding_size,
                               stride=stride)

        self.batch_norm = batch_norm
        if batch_norm:
            self.batch_norm_layer = nn.BatchNorm2d(out_channels)
        self.size_fix = None
        if in_channels != out_channels:
            self.size_fix = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        residual = inputs
        x = self.conv1(inputs)
        x = self.batch_norm_layer(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm_layer(x)
        if self.size_fix:
            residual = self.size_fix(residual)
        return self.relu(x + residual)


class Sum(nn.Module):
    def __init__(self, channels):
        super(Sum, self).__init__()
        self.extra_block = None
        self.channels = channels
        self.extra_block = nn.Sequential(
            nn.Conv2d(in_channels=self.channels[0], out_channels=self.channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channels[1]))

    def forward(self, inputs1, inputs2):

        difference = inputs1.size()[2] - inputs2.size()[2]

        if difference < 0:
            inputs1 = F.pad(inputs1, [(math.ceil(abs(difference / 2))), (math.floor(abs(difference / 2))),
                                      (math.ceil(abs(difference / 2))), (math.floor(abs(difference / 2)))])
        elif difference > 0:
            inputs2 = F.pad(inputs2, [(math.ceil(abs(difference / 2))), (math.floor(abs(difference / 2))),
                                      (math.ceil(abs(difference / 2))), (math.floor(abs(difference / 2)))])

        if self.channels[0] != self.channels[1]:
            inputs1 = self.extra_block(inputs1)

        return inputs1 + inputs2


class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, inputs1, inputs2):
        difference = inputs1.size()[2] - inputs2.size()[2]

        if difference < 0:
            inputs1 = F.pad(inputs1, [(math.ceil(abs(difference / 2))), (math.floor(abs(difference / 2))),
                                      (math.ceil(abs(difference / 2))), (math.floor(abs(difference / 2)))])
        elif difference > 0:
            inputs2 = F.pad(inputs2, [(math.ceil(abs(difference / 2))), (math.floor(abs(difference / 2))),
                                      (math.ceil(abs(difference / 2))), (math.floor(abs(difference / 2)))])

        return torch.cat([inputs1, inputs2], dim=1)


class ConvNet(nn.Module):

    def __init__(self, decoded_chromosome, channel=3, n_class=200, input_size=64):
        super(ConvNet, self).__init__()
        self.chromosome = decoded_chromosome
        size = input_size
        channel_size = channel
        layers = []
        for layer in decoded_chromosome:
            layer = layer[0]
            name = layer[0]
            if name == "max" or name == "avg":
                size = size // 2
                if name == "max":
                    layers.append(nn.MaxPool2d(2, 2))
                else:
                    layers.append(nn.AvgPool2d(2, 2))
            elif name == "conv":
                layers.append(ConvBlock(in_channels=channel_size, out_channels=layer[1], kernel_size=layer[2]))
                channel_size = layer[1]
            elif name == "res":
                layers.append(ResBlock(in_channels=channel_size, out_channels=layer[1], kernel_size=layer[2]))
                channel_size = layer[1]
            if size < 1:
                raise ValueError("Bad Network")

        layers.append(nn.Linear(channel_size * size * size, n_class))
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs):
        output = inputs
        for index, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                return F.sigmoid(
                    layer(output.view(output.size(0), -1))
                )
            elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
                output = (layer(output))
            elif isinstance(layer, ConvBlock) or isinstance(layer, ResBlock):
                output = (layer(output))