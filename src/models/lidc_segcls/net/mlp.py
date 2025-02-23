from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size: int, 
                 hidden_size_list: List[int], 
                 n_classes: Union[int, List[int]],  # Hỗ trợ cả single-task và multi-head
                 dropout_rate: float = 0.5) -> None:
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_size_list[0]))
        layers.append(nn.BatchNorm1d(hidden_size_list[0]))  # Thêm BatchNorm
        layers.append(nn.LeakyReLU())  # Dùng LeakyReLU thay vì ReLU
        layers.append(nn.Dropout(dropout_rate))  

        for i in range(1, len(hidden_size_list)):
            layers.append(nn.Linear(hidden_size_list[i - 1], hidden_size_list[i]))
            layers.append(nn.BatchNorm1d(hidden_size_list[i]))  # Thêm BatchNorm
            layers.append(nn.LeakyReLU())  
            layers.append(nn.Dropout(dropout_rate))  

        self.layers = nn.Sequential(*layers)

        # Hỗ trợ cả single-task (n_classes là int) và multi-task (n_classes là List[int])
        if isinstance(n_classes, int):
            self.heads = nn.Linear(hidden_size_list[-1], n_classes)
        else:
            self.heads = nn.ModuleList([nn.Linear(hidden_size_list[-1], num_label_or_class) for num_label_or_class in n_classes])

    def forward(self, x):
        x = self.layers(x)
        
        if isinstance(self.heads, nn.Linear):
            return self.heads(x)  # Single-task classification
        
        outputs = [head(x) for head in self.heads]  # Multi-task classification
        return outputs

if __name__ == "__main__":
    mlp = MLP(input_size=32,
              hidden_size_list=[16, 8],
              n_classes=[4, 7],  # Hoặc một số nguyên nếu single-task
              dropout_rate=0.5)

    input_data = torch.randn(10, 32)
    outputs = mlp(input_data)

    if isinstance(outputs, list):
        for out in outputs:
            print(out.shape)
    else:
        print(outputs.shape)
