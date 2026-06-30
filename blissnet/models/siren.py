import math
import torch
import torch.nn as nn

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights(is_first)

    def _init_weights(self, is_first):
        with torch.no_grad():
            if is_first:
                bound = 1.0 / self.linear.in_features
            else:
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SIRENTrunk(nn.Module):
    def __init__(self, coord_dim=2, hidden_width=512, K=512, omega_0=30.0):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            SineLayer(coord_dim, hidden_width, omega_0=omega_0, is_first=True),
            SineLayer(hidden_width, hidden_width, omega_0=omega_0),
            SineLayer(hidden_width, hidden_width, omega_0=omega_0),
            nn.Linear(hidden_width, K),
        )
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_width) / omega_0
            self.net[-1].weight.uniform_(-bound, bound)
            self.net[-1].bias.uniform_(-bound, bound)

    def forward(self, coords):
        return self.net(coords)