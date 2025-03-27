import torch
import torch.nn as nn
from typing import List

class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)  # Residual connection

class ResNet1D(nn.Module):
    def __init__(self, input_size: int, 
                 hidden_size_list: List[int], 
                 n_classes: List[int], 
                 dropout_rate: float = 0.5):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_size_list[0]))
        layers.append(nn.BatchNorm1d(hidden_size_list[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        for i in range(1, len(hidden_size_list)):
            layers.append(ResidualBlock(hidden_size_list[i-1]))
            layers.append(nn.Linear(hidden_size_list[i-1], hidden_size_list[i]))
            layers.append(nn.BatchNorm1d(hidden_size_list[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        self.feature_extractor = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(hidden_size_list[-1], out) for out in n_classes])

    def forward(self, x):
        x = self.feature_extractor(x)
        outputs = [head(x) for head in self.heads]
        return outputs

if __name__ == "__main__":
    model = ResNet1D(input_size=32,
                     hidden_size_list=[64, 32],
                     n_classes=[4, 7],
                     dropout_rate=0.5)

    x = torch.randn(10, 32)  # batch x features
    outs = model(x)

    for out in outs:
        print(out.shape)
