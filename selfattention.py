import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class SelfAttLayer(nn.Module):
    def __init__(self, d):
        super(SelfAttLayer, self).__init__()
        self.d = d
        self.WQ= nn.Parameter(torch.randn(d,d))
        self.Wk = nn.Parameter(torch.randn(d, d))
        self.Wv = nn.Parameter(torch.randn(d, d))
    def forward(self, text):
        Q=torch.matmul(text,self.WQ)
        K=torch.matmul(text,self.Wk)
        V=torch.matmul(text,self.Wv)
        H=torch.bmm(K.permute(0,2,1),Q)/self.d**0.5
        H=F.softmax(H)
        print(H.size())
        H=torch.bmm(V,H)
        return H
if __name__ == '__main__':
    pass



