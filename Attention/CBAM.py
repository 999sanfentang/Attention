import torch
from torch import nn


class channal_attention(nn.Module):
    def __init__(self, channel, radio=16):
        super(channal_attention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel//radio, False),
            nn.ReLU(),
            nn.Linear(channel//radio, channel, False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, w, h = x.size()
        # b,c,w,h -> b,c,1,1 -> b,c
        max_pool_out = self.max_pool(x).view([b, c])
        avg_pool_out = self.avg_pool(x).view([b, c])
        # b,c -> b, c//radio -> b,c
        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)

        out = max_fc_out + avg_fc_out  # 元素级别相加
        # b,c -> b,c,1,1
        out = self.sigmoid(out).view([b, c, 1, 1])

        return x * out


class spacial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spacial_attention, self).__init__()

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, 1, padding = padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_pool_out = torch.mean(x, dim=1, keepdim=True)
        pool_out = torch.cat([max_pool_out,mean_pool_out], dim=1)
        out = self.conv(pool_out)
        out = self.sigmoid(out)
        return out * x


class CBAM(nn.Module):
    def __init__(self, channel, radio=16, kernel_size=7):
        super(CBAM,self).__init__()
        self.channel_attention = channal_attention(channel, radio)
        self.spacial_attention = spacial_attention(kernel_size)
        
    def forward(self,x):
        b, c, w, h = x.size()
        channal_attention_out = self.channel_attention(x)
        spacial_attention_out = self.spacial_attention(channal_attention_out)
        
        return spacial_attention_out


model = CBAM(512)
print(model)
inputs = torch.ones([2, 512, 26, 26])
outputs = model(inputs)
