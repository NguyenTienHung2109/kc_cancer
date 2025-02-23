from typing import List
import torch
import torch.nn as  nn

class MLP(nn.Module):

    def __init__(self, input_size: int, 
                hidden_size_list: List[int], 
                n_classes: List[int],
                dropout: float = 0.1) -> None:
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_size_list[0]))
        layers.append(nn.ReLU()) 
        layers.append(nn.Dropout(dropout))

        for i in range(1, len(hidden_size_list)):
            layers.append(nn.Linear(hidden_size_list[i-1], hidden_size_list[i]))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(hidden_size_list[-1], num_label_or_class) for num_label_or_class in n_classes])
        self.n_classes = n_classes

    def forward(self, x):
        x = self.layers(x)
        outputs = [head(x) for head in self.heads]
        return outputs

if __name__ == "__main__":
    mlp = MLP(input_size=1024,
            hidden_size_list=[256, 64, 16],
            n_classes=[4, 8, 3, 5, 4])

    input = torch.randn(10, 1024)
    outputs = mlp(input)

    for out in outputs:
        print(out.shape)