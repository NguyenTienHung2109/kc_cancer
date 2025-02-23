from typing import Optional, Callable
import torch
import torch.nn as  nn
import torch.nn.functional as F
    
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(BasicBlock, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm2(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


        
        
class ResNet(nn.Module):
    def __init__(self, 
                 n_labels_or_classes, 
                 num_channels=3, 
                 channels=64,
                 architecture: str = "resnet50"):
        super(ResNet, self).__init__()

        if architecture == "resnet18":
            layer_list = [2, 2, 2, 2]
            ResBlock = BasicBlock
        elif architecture == "resnet34":
            layer_list = [3, 4, 6, 3]
            ResBlock = BasicBlock
        elif architecture == "resnet50":
            layer_list = [3, 4, 6, 3]
            ResBlock = Bottleneck
        elif architecture == "resnet51":
            layer_list = [4, 6, 3]
            ResBlock = Bottleneck
        elif architecture == "resnet101":
            layer_list = [3, 4, 23, 3]
            ResBlock = Bottleneck
        elif architecture == "resnet152":
            layer_list = [3, 8, 36, 3]
            ResBlock = Bottleneck
        else:
            raise NotImplementedError(f"not implemented {architecture}")

        self.in_channels = channels
        
        self.conv1 = nn.Conv2d(num_channels, channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        channels = channels
        self.layers = []
        for i, n_block in enumerate(layer_list):
            layer = self._make_layer(ResBlock, 
                                     n_block, 
                                     planes=channels,
                                     stride=1 if i == 0 else 2)
            self.layers.append(layer)
            channels = channels << 1
        channels = channels >> 1

        self.layers = nn.Sequential(*self.layers)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.heads = nn.ModuleList([nn.Linear(channels*ResBlock.expansion, num_label_or_class) for num_label_or_class in n_labels_or_classes])
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        x = self.layers(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        
        outputs = [head(x) for head in self.heads]
        return outputs
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

if __name__ == "__main__":
    resnet = ResNet(n_labels_or_classes=[3, 7, 1, 4, 3], 
                    num_channels=1,
                    channels=32,
                    architecture="resnet50")

    input = torch.randn(10, 1, 256, 256)
    outputs = resnet(input)

    for out in outputs:
        print(out.shape)