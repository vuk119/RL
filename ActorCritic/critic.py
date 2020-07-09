import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """
    Takes in the state. Outputs the value state function.

    Input size: observation space size
    Output size: 1
    """

    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.input_layer = nn.Linear(input_size, 64)
        self.fc = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.fc(x))
        x = self.output_layer(x)

        return x
