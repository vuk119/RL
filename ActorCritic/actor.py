import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.input_layer = nn.Linear(input_size, 64)
        self.fc = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.fc(x))
        x = self.output_layer(x)

        return x
