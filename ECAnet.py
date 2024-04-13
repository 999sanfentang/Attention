import torch
from torch import nn
import math
from torchstat import stat  # 查看网络参数



class eca_block(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(in_channel, 2)+b)/gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size // 2
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        avg = self.avg_pool(x).view([b, 1, c])
        out = self.conv(avg)
        print(out.shape)
        out = self.sigmoid(out).view([b, c, 1, 1])
        print(out.shape)
        
        return out * x
    

# model = eca_block(512)
# print(model)
# inputs = torch.ones([2, 512, 26, 26])
# outputs = model(inputs)
# 构造输入层 [b,c,h,w]==[4,32,16,16]
inputs = torch.rand([4, 32, 16, 16])
# 获取输入图像的通道数
in_channel = inputs.shape[1]
# 模型实例化
model = eca_block(in_channel=in_channel)
# 前向传播
outputs = model(inputs)

print(outputs.shape)  # 查看输出结果
print(model)    # 查看网络结构
stat(model, input_size=[32, 16,16])  # 查看网络参数