import torch
from torch import nn

# 在DAF_CA的基础上，再添加一条空间注意力分支
#----------------------------------------#
class spacial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spacial_attention, self).__init__()

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, 1,
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # b, c, h, w => b,1,h,w
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_pool_out = torch.mean(x, dim=1, keepdim=True)
        # b,2,h,w
        pool_out = torch.cat([max_pool_out, mean_pool_out], dim=1)
        out = self.conv(pool_out)
        out = self.sigmoid(out)
        return out
# ----------------------------------------#

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class myTCA_02(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(myTCA_02, self).__init__()
        mip = max(8, inp // reduction)

        # 卷积层，将输入通道数量从inp减少到mip，大小从inp * h * w变为mip * h * w
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=(
            1, 2), stride=1, padding=(0, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(mip)  # 批归一化层，对mip个通道进行归一化
        self.act = h_swish()  # 激活函数
        # 高度卷积层，将通道数量从mip扩大到oup
        self.conv_h = nn.Conv2d(mip, oup, 1, 1, bias=False)
        # 宽度卷积层，将通道数量从mip扩大到oup
        self.conv_w = nn.Conv2d(mip, oup, 1, 1, bias=False)
        
        self.spacial_attention = spacial_attention(kernel_size=7)

    def forward(self, x):
        identity = x  # 保存原始输入，形状为(b, inp, h, w)
        _, _, h, w = x.size()

        pool_ha = nn.AdaptiveAvgPool2d((h, 1))  # 平均池化，结果形状为(b, inp, h, 1)
        pool_hm = nn.AdaptiveMaxPool2d((h, 1))  # 最大池化，结果形状为(b, inp, h, 1)
        x_ha = pool_ha(x)
        x_hm = pool_hm(x)
        x_h = torch.cat([x_ha, x_hm], dim=3)  # 沿宽度方向拼接，结果形状为(b, inp, h, 2)

        pool_wa = nn.AdaptiveAvgPool2d((1, w))  # 结果形状为(b, inp, 1, w)
        pool_wm = nn.AdaptiveMaxPool2d((1, w))  # 结果形状为(b, inp, 1, w)
        x_wa = pool_wa(x).permute(0, 1, 3, 2)  # 维度置换后，结果形状为(b, inp, w, 1)
        x_wm = pool_wm(x).permute(0, 1, 3, 2)  # 维度置换后，结果形状为(b, inp, w, 1)
        x_w = torch.cat([x_wa, x_wm], dim=3)  # 沿高度方向拼接，结果形状为(b, inp, w, 2)

        y1 = torch.cat([x_h, x_w], dim=2)  # 沿高度方向拼接，结果形状为(b, inp, h+w, 2)
        # print(y1.shape)  # torch.Size([2, 32, 52, 2])
        y1 = self.conv1(y1)  # 卷积操作，(b, inp, h+w, 2) => (b, mip, h+w, 1)
        # print(y1.shape)  # torch.Size([2, 8, 52, 1])
        y1 = self.bn1(y1)
        y1 = self.act(y1)

        # 拆分，形状的变化为(b, mip, h, 1) 和 (b, mip, w, 1)
        y_h, y_w = torch.split(y1, [h,  w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)  # y_w维度置换，形状变为(b, mip, 1, w)
        a_h = (self.conv_h(y_h)).sigmoid()  # 形状变为(b, oup, h, 1)
        a_w = (self.conv_w(y_w)).sigmoid()  # 形状变为(b, oup, 1, w)
        
        spacial_attention = self.spacial_attention(x)

        return identity * a_h * a_w * spacial_attention  # 结果形状为(b, oup, h, w)


# Define your parameters
inp = 32  # the number of input channels
oup = 32  # the number of output channels
reduction = 16  # reduction factor for mip calculation

# Create instance of DAF_CA
daf_ca_instance = myTCA_02(inp, oup, reduction)

print(daf_ca_instance)
inputs = torch.ones([2, 32, 26, 26])
outputs = daf_ca_instance(inputs)
# Now you can use daf_ca_instance to process your inputs
