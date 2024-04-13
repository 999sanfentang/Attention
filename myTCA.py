import torch
from torch import nn
from torchstat import stat  # 查看网络参数s


class spacial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spacial_attention, self).__init__()

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, 1,
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # 使用最大池化获取每个通道的最大值，并保持维度不变，维度是(b, c, h, w) => (b, 1, h, w)
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        # 使用平均池化获取每个通道的平均值，并保持维度不变，维度是(b, c, h, w) => (b, 1, h, w)
        mean_pool_out = torch.mean(x, dim=1, keepdim=True)
        # (b, 1, h, w) => (b, 2, h, w)
        pool_out = torch.cat([max_pool_out, mean_pool_out], dim=1)
        # (b, 2, h, w) => (b, 1, h, w)
        out = self.conv(pool_out)
        # 生成权重
        out = self.sigmoid(out)
        # print(out)
        return out


class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(CA_Block, self).__init__()
        self.spacial_attention = spacial_attention(kernel_size)

        self.con_1x1 = nn.Conv2d(
            channel, channel//reduction, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(in_channels=channel//reduction,
                             out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction,
                             out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, 1,
                              padding=padding, bias=False)

    def forward(self, x):
        # 输入x的形状是 (b, c, h, w)
        _, _, h, w = x.size()
        # 对输入特征图在宽度方向上求平均，得到形状为(b, c, h, 1)的特征图，
        # 通过permute变换，得到形状为(b, c, 1, h) 的特征图x_h
        x_h_mean = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_h_max, _ = torch.max(x, dim=3, keepdim=True)
        x_h_max = x_h_max.permute(0, 1, 3, 2)
        # b,c,1,h cat b,c,1,h => b,c,2,h
        x_h_out = torch.cat([x_h_max, x_h_mean], dim=2)

        # (b, c, h, w) => (b, c, 1, w)
        # (b, c, h, w) => (b, c, 1, w)
        # (b, c, 1, w) cat (b, c, 1, w) => (b, c, 2, w)
        x_w_mean = torch.mean(x, dim=2, keepdim=True)
        x_w_max, _ = torch.max(x, dim=2, keepdim=True)
        x_w_out = torch.cat([x_w_mean, x_w_max], dim=2)

        # b, c, 2, w cat b, c, 2, h => b, C, 2, W + h => b, 2, c, W + h =>b, 1, c, W + h
        x_cat = self.conv(torch.cat((x_h_out, x_w_out), 3).permute(0, 2, 1, 3))

        # b, 1, c, W + h => b, c, 1, W + h => b, c/r, 1, w + h
        x_cat_conv_relu = self.relu(
            self.bn(self.con_1x1(x_cat.permute(0, 2, 1, 3))))
        # b, c / r, 1, w + h => b, c / r, 1, h and b, c / r, 1, W
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([
                                                                       h, w], 3)
        # b, c / r, 1, h => b, c / r, h, 1 => b, c, h, 1
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        # b, c / r, 1, W => b, c, 1, W
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        s_space = self.spacial_attention(x)
        out = x * s_h.expand_as(x) * s_w.expand_as(x) * s_space
        # print(s_space.shape)
        # print(s_h.expand_as(x).shape)
        # print(s_w.expand_as(x).shape)
        # out1 = x * s_space
        # print(out1.shape)
        # print(out.shape)

        return out


model = CA_Block(32)
print(model)
inputs = torch.ones([2, 32, 26, 26])
outputs = model(inputs)
stat(model, input_size=[32, 26, 26])  # 查看网络参数
