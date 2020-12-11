import torch
import torch.nn as nn
import numpy as np
from objectives import cca_loss


class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size):
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes  # adding input size to layer list
        for l_id in range(len(layer_sizes) - 1):  # travesing the list to add layer data
            if l_id == len(layer_sizes) - 2: # if not last layer
                layers.append(nn.Sequential(
                    nn.BatchNorm1d(num_features=layer_sizes[l_id], affine=False), # batchnorm is used to stabalize the network
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                ))
            else: # last layer
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.Sigmoid(), # activation function
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1], affine=False),
                ))
        self.layers = nn.ModuleList(layers) # combining linear layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Model(nn.Module):
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2, outdim_size, use_all_singular_values, device=torch.device('cpu')):
        super(Model, self).__init__()
        self.model1 = MlpNet(layer_sizes1, input_size1).double()  # passing input X to neural network 1 
        self.model2 = MlpNet(layer_sizes2, input_size2).double()  # passing input Y to neural network 2

        self.loss = cca_loss(outdim_size, use_all_singular_values, device).loss # loss for backpropogaiton

    def forward(self, x1, x2):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        # feature * batch_size
        output1 = self.model1(x1)
        output2 = self.model2(x2)

        return output1, output2
