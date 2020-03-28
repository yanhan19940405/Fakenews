import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class AttLayer(nn.Module):
    def __init__(self, output_size):
        super(AttLayer, self).__init__()
        self.output_size = output_size
        self.linear=torch.nn.Linear(output_size,output_size)
    def forward(self, text):
        Q=self.linear(text).float().cpu().detach().numpy()
        K=self.linear(text).float().cpu().detach().numpy()
        V=self.linear(text).float().cuda()
        H=[]
        for i in range(len(K)):
            H.append(np.dot(Q[i],K[i].T)/self.output_size**0.5)
        H=torch.from_numpy(np.array(H)).float()
        H=F.softmax(H).cuda()
        H=torch.bmm(H,V)
        return H
if __name__ == '__main__':
    pass



