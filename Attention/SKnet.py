'''
Descripttion: 
Result: 
Author: Philo
Date: 2023-03-02 14:55:44
LastEditors: Philo
LastEditTime: 2023-03-02 16:01:03
'''
import torch.nn as nn
import torch


class SKConv(nn.Module):
    def __init__(self, in_ch, M=3, G=1, r=4, stride=1, L=32) -> None:
        super().__init__()
        """ Constructor
        Args:
        in_ch: input channel dimensionality.输入通道维度。
        M: the number of branchs.分支机构的数量。
        G: num of convolution groups.卷积群的个数。
        r: the radio for compute d, the length of z.计算d的无线电，z的长度。
        stride: stride, default 1.步幅，默认为1。
        L: the minimum dim of the vector z in paper, default 32.纸张中矢量z的最小亮度，默认值为32
        """
        d = max(int(in_ch/r), L)  # 用来进行线性层的输出通道，当输入数据In_ch很大时，用L就有点丢失数据了。
        self.M = M
        self.in_ch = in_ch
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, kernel_size=3+i*2,
                              stride=stride, padding=1+i, groups=G),
                    nn.BatchNorm2d(in_ch),
                    nn.ReLU(inplace=True)
                )
            )
        # print("D:", d)
        self.fc = nn.Linear(in_ch, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, in_ch))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 第一部分，每个分支的数据进行相加,虽然这里使用的是torch.cat，但是后面又用了unsqueeze和sum进行升维和降维
        for i, conv in enumerate(self.convs):
            # 这里在1这个地方新增了一个维度  16*1*64*256*256
            fea = conv(x).clone().unsqueeze_(dim=1).clone()
            if i == 0:
                feas = fea
            else:
                # feas.shape  batch*M*in_ch*W*H
                feas = torch.cat([feas.clone(), fea], dim=1)
        fea_U = torch.sum(feas.clone(), dim=1)  # batch*in_ch*H*W
        fea_s = fea_U.clone().mean(-1).mean(-1)  # Batch*in_ch
        fea_z = self.fc(fea_s)  # batch*in_ch-> batch*d
        for i, fc in enumerate(self.fcs):
            # print(i, fea_z.shape)
            # batch*d->batch*in_ch->batch*1*in_ch
            vector = fc(fea_z).clone().unsqueeze_(dim=1)
            # print(i, vector.shape)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat(
                    [attention_vectors.clone(), vector], dim=1)  # 同样的相加操作 # batch*M*in_ch
        attention_vectors = self.softmax(
            attention_vectors.clone())  # 对每个分支的数据进行softmax操作
        attention_vectors = attention_vectors.clone(
        ).unsqueeze(-1).unsqueeze(-1)  # ->batch*M*in_ch*1*1
        # ->batch*in_ch*W*H
        fea_v = (feas * attention_vectors).clone().sum(dim=1)
        return fea_v


if __name__ == "__main__":
    x = torch.randn(16, 64, 256, 256)
    sk = SKConv(in_ch=64,  M=3, G=1, r=2)
    out = sk(x)
    print(out.shape)
    # in_ch 数据输入维度，M为分指数，G为Conv2d层的组数，基本设置为1，r用来进行求线性层输出通道的。


