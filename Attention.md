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



## 6 DAF_CA

![image-20240415150000489](https://typoraa001.oss-cn-beijing.aliyuncs.com/image-20240415150000489.png)

在CA基础上添加max池化





## 7  ELA

EfficientLocalizationAttention

![image-20240415111545728](https://typoraa001.oss-cn-beijing.aliyuncs.com/image-20240415111545728.png)

1.用GN代替BN

2.用1d卷积 取消通道降维



