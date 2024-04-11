import torch 
from torch import nn

class senet(nn.Module):
    def __init__(self,channel,radio = 16):
        super(senet,self).__init__()
        self.ave_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//radio, False),
            nn.ReLU(),
            nn.Linear(channel//radio, channel,False),
            nn.Sigmoid(),
        )
        
    def forward(self,x):
        b,c,h,w = x.size()
        # b,c,,h,w -> b,c,1,1
        avg = self.ave_pool(x).view([b,c])
        # b,c -> b,c//radio -> b,c -> b,c,1,1
        fc = self.fc(avg).view([b, c, 1, 1])
        
        print(fc)
        return x*fc

model = senet(512)   
print(model) 
inputs = torch.ones([2,512,26,26])
outputs = model(inputs)
           