from torchstat import stat  # 查看网络参数
import torch
from torch import nn

# 空间注意力
class ZPool(nn.Module):
    def forward(self, x):
        b, c, w, h = x.size()
        # 使用最大池化获取每个通道的最大值，并保持维度不变，维度是(b, c, h, w) => (b, 1, h, w)
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        # 使用平均池化获取每个通道的平均值，并保持维度不变，维度是(b, c, h, w) => (b, 1, h, w)
        mean_pool_out = torch.mean(x, dim=1, keepdim=True)
        # 拼接后的维度为(b, 2, h, w)
        z_pool = torch.cat((max_pool_out, mean_pool_out), dim=1)

        return z_pool

# conv+cn+relu
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1,
                              padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        # b,c,h,w => b,h,c,w
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        # b,h,c,w => b,2*h,c,w => b,h,c,w
        # zpool -> (conv+bn+relu)+sigmiod
        x_out1 = self.cw(x_perm1)
        # b,h,c,w => b,c,h,w
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        
        # b,c,h,w => b,w,h,c
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        # b,w,h,c => b,2*w,h,c => b,w,h,c
        # zpool => (conv+bn+relu)+sigmiod
        x_out2 = self.hc(x_perm2)
        # b,w,h,c => b,c,h,w
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        
        
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


inputs = torch.rand([4, 32, 16, 16])
# 获取输入图像的通道数
in_channel = inputs.shape[1]
# 模型实例化
model = TripletAttention()
# 前向传播
outputs = model(inputs)

print(outputs.shape)  # 查看输出结果
print(model)    # 查看网络结构
stat(model, input_size=[32, 16, 16])  # 查看网络参数