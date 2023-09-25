from torch import nn

class Net(nn.Module):
    """
    A PyTorch Module for a fully connected neural network.
    """

    def __init__(self, n_layers=3, n_units=64, dropout_rate=0.2, input_shape=35, activation=nn.ReLU(), output_units=1, output_activation=nn.Sigmoid()):
        """
        Initializes the Net object with the given parameters.
        """
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
        """
        Defines the computation performed at every call.
        """
        for layer in self.layers:
            x = layer(x)
        return x
