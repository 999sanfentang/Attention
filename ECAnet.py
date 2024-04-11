import torch
from torch import nn
import math


class eca_block(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2)+b)/gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size // 2
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        avg = self.avg_pool(x).view([b, 1, c])
        out = self.conv(avg)
        out = self.sigmoid(out).view([b, c, 1, 1])
        
        return out * x
    

model = eca_block(512)
print(model)
inputs = torch.ones([2, 512, 26, 26])
outputs = model(inputs)
