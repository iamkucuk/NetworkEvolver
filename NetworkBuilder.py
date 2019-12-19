import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="same", activation="relu",
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


# class SumAfterConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding="same", activation="relu",
#                  batch_norm=True):
#         super(SumAfterConv, self).__init__()
#
#         padding_size = 0
#
#         if padding == "same":
#             padding_size = kernel_size // 2
#
#         self.layer = nn.Conv2d(in_channels=in_channels,
#                                out_channels=out_channels,
#                                kernel_size=kernel_size,
#                                padding=padding_size,
#                                stride=stride)
#
#         self.batch_norm = batch_norm
#
#         if batch_norm:
#             self.batch_norm_layer = nn.BatchNorm2d(out_channels)
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, inputs1, inputs2):
#         x = self.layer(inputs1)
#         if self.batch_norm:
#             x = self.batch_norm_layer(x)
#         x = F.relu(x)
#
#         difference = x.size()[2] - inputs2.size()[2]
#
#         if difference < 0:
#             x = F.pad(x, [(math.ceil(abs(difference / 2))), (math.floor(abs(difference / 2)))])
#         elif difference > 0:
#             inputs2 = F.pad(inputs2, [(math.ceil(abs(difference / 2))), (math.floor(abs(difference / 2)))])
#
#         return self.relu(x + inputs2)


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
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        residual = inputs
        x = self.conv1(inputs)
        if self.batch_norm:
            x = self.batch_norm_layer(x)
        x = self.relu(x)
        x = self.conv2(inputs)
        if self.batch_norm:
            x = self.batch_norm_layer(x)

        return self.relu(x + residual)


class Sum(nn.Module):
    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, inputs1, inputs2):

        difference = inputs1.size()[2] - inputs2.size()[2]

        if difference < 0:
            inputs1 = F.pad(inputs1, [(math.ceil(abs(difference / 2))), (math.floor(abs(difference / 2)))])
        elif difference > 0:
            inputs2 = F.pad(inputs2, [(math.ceil(abs(difference / 2))), (math.floor(abs(difference / 2)))])

        difference = inputs1.size()[1] - inputs2.size()[1]
        if difference < 0:
            inputs1 = torch.cat(
                [inputs1, torch.zeros((inputs1.shape[0], abs(difference), inputs1.shape[2], inputs1.shape[3]))],
                1)
        elif difference > 0:
            inputs2 = torch.cat(
                [inputs2, torch.zeros((inputs1.shape[0], abs(difference), inputs1.shape[2], inputs1.shape[3]))],
                1)

        return inputs1 + inputs2


class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, inputs1, inputs2):
        difference = inputs1.size()[2] - inputs2.size()[2]

        if difference < 0:
            inputs1 = F.pad(inputs1, [(math.ceil(abs(difference / 2))), (math.floor(abs(difference / 2)))])
        elif difference > 0:
            inputs2 = F.pad(inputs2, [(math.ceil(abs(difference / 2))), (math.floor(abs(difference / 2)))])

        return torch.cat([inputs1 + inputs2], 1)


class ConvNet(nn.Module):
    def __init__(self, decoded_chromosome, channel=3, n_class=100, input_size=64):
        super(ConvNet, self).__init__()
        self.chromosome = decoded_chromosome
        layers = []
        channels = []
        sizes = []
        channels.append(channel)
        sizes.append(input_size)
        size = (0, 0)
        self.inputs = []
        for layer, input1, input2 in decoded_chromosome:
            name = layer[0]
            if name == "dense":
                layers.append(nn.Linear(channel * size * size, n_class))
            elif name == "max" or name == "avg":
                channels.append(channels[input1])
                size = size / 2
                if name == "max":
                    layers.append(nn.MaxPool2d(2, 2))
                else:
                    layers.append(nn.AvgPool2d(2, 2))
            elif name == "concat":
                channels.append(channels[input1] + channels[input2])
                sizes.append(min(channels[input1], channels[input2]))
                layers.append(Concat())
            elif name == "sum":
                channels.append(max(channels[input1], channels[input2]))
                sizes.append(min(sizes[input1], sizes[input2]))
                layers.append(Sum())
            elif name == "conv":
                channels.append(layer[1])
                layers.append(ConvBlock(in_channels=channels[input1], out_channels=layer[1], kernel_size=layer[2]))
                sizes.append(sizes[input1])
                #size after padding will be added
            elif name == "res":
                channels.append(max(layer[1], channels[input1]))
                sizes.append(sizes[input1])
                layers.append(ResBlock(in_channels=channels[input1], out_channels=layer[1], kernel_size=layer[2]))
            self.inputs.append((input1, input2))
            if sizes[-1] < 1:
                raise Exception("Bad Network")

        self.layers = layers

    def forward(self, inputs):
        outputs = [inputs]
        for index, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                return F.sigmoid(
                    layer(outputs[self.inputs[index][0]].view(outputs[self.inputs[index][0]].size(0), -1))
                )
            elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
                if outputs[self.inputs[index][0]].size(2) > 1:
                    outputs.append(layer(outputs[self.inputs[index][0]]))
                else:
                    outputs.append(outputs[self.inputs[index][0]])
            elif isinstance(layer, ConvBlock) or isinstance(layer, ResBlock):
                outputs.append(layer(outputs[self.inputs[index][0]]))
            elif isinstance(layer, Concat) or isinstance(layer, Sum):
                outputs.append(layer(outputs[self.inputs[index][0]], outputs[self.inputs[index][1]]))
