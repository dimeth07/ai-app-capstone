# src/models/model.py
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features=2, hidden_sizes=(8,), num_classes=2):
        super().__init__()
        # The layers are defined in __init__
        layers = []
        last_size = in_features
        for h_size in hidden_sizes:
            layers.append(nn.Linear(last_size, h_size))
            layers.append(nn.ReLU()) # We will change this later for the ablation
            last_size = h_size
        layers.append(nn.Linear(last_size, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # The data flow is defined in forward
        # The output is 'logits', not probabilities
        return self.net(x)