# Attention

## 1 SE

![image-20240411203623708](https://typoraa001.oss-cn-beijing.aliyuncs.com/image-20240411203623708.png)

通道注意力机制：

先经过一个保留通道维度的全局平均池化，然后经过全连接缩小--relu--全连接放大回原来--sigmiod输出权重，跟原来特征相乘（h，w变为1，1）

## 2 CBAM

![image-20240411203646462](https://typoraa001.oss-cn-beijing.aliyuncs.com/image-20240411203646462.png)

通道注意力---空间注意力

先是和SE一样，构造一个通道注意力，但是在SE基础上，除了平均池化，还加了一路最大池化。也就是说，将输入进来的特征进行保留通道维度的全局最大/平均池化（h，w变为1，1），然后分别进行FC层（缩小-relu-放大），之后两个长条相加，经过sigmoid，得到权重。跟原来的特征相乘。输出X1

将第一步通道注意力得到的，已经与权重成过之后的特征X1，进行保留空间维度（保留h，w），c通道留一层。最大 池化和平均池化，然后两个特征层相加，相加后过一个卷积，使得通道数变为1，然后经过sigmoid得到权重。与X1相乘得到最后的带权重的特征。

## 3 ECA

![image-20240411203655118](https://typoraa001.oss-cn-beijing.aliyuncs.com/image-20240411203655118.png)

作者表明 **SENet 中的降维会给通道注意力机制带来副作用**，并且**捕获所有通道之间的依存关系是效率不高的，而且是不必要的**。**通过** **一维卷积 layers.Conv1D** **来完成跨通道间的信息交互**

（1）将输入特征图经过全局平均池化，特征图从 [b,c,h,w] 的矩阵变成 [b,c,1,1] 的向量,在reshape成 [b,1,c] 

（2）根据特征图的通道数计算得到自适应的一维卷积核大小 kernel_size

（3）将 kernel_size 用于一维卷积中，得到对于特征图的每个通道的权重

（4）将归一化权重和原输入特征图逐通道相乘，生成加权后的特征图


## 4 CA

![image-20240411203601592](https://typoraa001.oss-cn-beijing.aliyuncs.com/image-20240411203601592.png)

全面关注特征层的空间信息和通道信息

注意力机制（如CBAM、SE）在求取通道注意力的时候，**通道的处理**一般是采用全局最大池化/平均池化，这样会损失掉物体的空间信息。作者期望在引入通道注意力机制的同时，引入空间注意力机制，作者提出的注意力机制将位置信息嵌入到了通道注意力中。

==CBAM穿行的串行的通道+空间 =》 并行的通道+空间==

将输入的特征图分别在宽度和高度两个方向进行全局平均池化，宽方向的平均池化permute一下，将立起来的特征图放倒然后和高方向的特征图在dim=3维度cat

然后经过一个1*1 2d卷积缩小通道数，然后经过BN+relu

再split成两个，h方向和w方向，也就是==还原==过程。h方向立起来，经过1*1 2d卷积还原通道数，经过sigmoid。w方向经过1 * 1 2d卷积还原通道数，经过sigmoid。将两个权重和原来的特征，三者相乘。

![image-20240411210445027](https://typoraa001.oss-cn-beijing.aliyuncs.com/image-20240411210445027.png)

## 5 TA（**triplet attention**）

![image-20240411213713458](https://typoraa001.oss-cn-beijing.aliyuncs.com/image-20240411213713458.png)

![image-20240411213900646](https://typoraa001.oss-cn-beijing.aliyuncs.com/image-20240411213900646.png)

第一个：空间注意力（和CBAM的空间注意力一样，在conv后加了一个BN）

第二个分支：交换h，c--》做第一步重复

第三个分支：交换w，c-》做第一步重复

三者相加求avg

感觉很水

![image-20240412165053038](https://typoraa001.oss-cn-beijing.aliyuncs.com/image-20240412165053038.png)

eca：不降维，不建立通道连接



ca、TA：多维度交互（cwh交互）



## 6  ELA

![image-20240415111545728](https://typoraa001.oss-cn-beijing.aliyuncs.com/image-20240415111545728.png)

```
class EfficientLocalizationAttention(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(EfficientLocalizationAttention, self).__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size,
                              padding=self.pad, groups=channel, bias=False)
        self.gn = nn.GroupNorm(16, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 初始张量形状 (b, c, h, w)
        
        b, c, h, w = x.size()

        # 处理高度维度
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)
        # x_h 张量形状现在为 (b, c, h, 1)

        # 处理宽度维度
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)
        # x_w 张量形状现在为 (b, c, 1, w)

        print(x_h.shape, x_w.shape)
        # 在两个维度上应用注意力, 最终形状回到 (b, c, h, w)
        return x * x_h * x_w


# 示例用法 ELABase(ELA-B)
if __name__ == "__main__":
    # 创建一个形状为 [batch_size, channels, height, width] 的虚拟输入张量
    dummy_input = torch.randn(2, 64, 32, 32)

    # 初始化模块
    ela = EfficientLocalizationAttention(
        channel=dummy_input.size(1), kernel_size=7)

    # 前向传播
    output = ela(dummy_input)
    # 打印出输出张量的形状，它将与输入形状相匹配。
    print(f"输出形状: {output.shape}")
```

```
import torch
from torch import nn

# 在DAF_CA的基础上，再添加一条空间注意力分支
# ----------------------------------------#


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
    def __init__(self, inp, oup, reduction=32,kernel_size = 7):
        super(myTCA_02, self).__init__()
        mip = max(8, inp // reduction)
        mip02 = inp

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
        self.conv01 = nn.Conv2d(inp, mip02, kernel_size=(
            1, 2), stride=1, padding=(0, 0), bias=False)
        self.pad = kernel_size // 2
        self.conv02 = nn.Conv1d(inp, inp, kernel_size=kernel_size,
                               padding=self.pad, groups=inp, bias=False)

    def forward(self, x):
        identity = x  # 保存原始输入，形状为(b, inp, h, w)
        b, c, h, w = x.size()

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
        y2 = self.conv01(y1).view(b,c,h+w)  # torch.Size([2, 32, 52, 1]) => b,c,h+w
        y2 = self.conv02(y1).view(b, c, h+w, 1)  # 1d卷积 => b,c,h+w,1
        y2 = self.bn1(y2) # BN
        y2 = self.act(y2) # relu
        
        # # 拆分，形状的变化为(b, c, h, 1) 和 (b, c, w, 1)
        # y_h, y_w = torch.split(y2, [h,  w], dim=2)
        # y_w = y_w.permute(0, 1, 3, 2)  # y_w维度置换，形状变为(b, c, 1, w)
        # a_h = y_h.sigmoid()
        # a_w = y_w.sigmoid()
        # print(a_h.shape)
        # print(a_w.shape)
        
        
        y1 = self.conv1(y1)  # 卷积操作，(b, inp, h+w, 2) => (b, mip, h+w, 1)
        # print(y1.shape)  # torch.Size([2, 8, 52, 1])
        y1 = self.bn1(y1)
        y1 = self.act(y1)

        # 拆分，形状的变化为(b, mip, h, 1) 和 (b, mip, w, 1)
        y_h, y_w = torch.split(y2, [h,  w], dim=2)
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

```

