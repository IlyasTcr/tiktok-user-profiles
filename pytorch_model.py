import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

class Net(nn.Module):
    def __init__(self, n_layers=3, n_units=64, dropout_rate=0.2, input_shape=35, activation=nn.ReLU(), output_units=1, output_activation=nn.Sigmoid()):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        
        for _ in range(n_layers):
            self.layers.append(nn.Linear(input_shape, n_units))
            self.layers.append(activation)
            self.layers.append(nn.Dropout(dropout_rate))
            input_shape = n_units
            
        self.layers.append(nn.Linear(n_units, output_units))
        self.layers.append(output_activation)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x