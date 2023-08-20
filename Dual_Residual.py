import torch
import torch.nn as nn
class Dual_Residual_Block(nn.Module):
    def __init__(self,dim,norm):
        super(Dual_Residual_Block,self).__init__()
        self.norm = norm
        self.dim = dim
    def forward(self,x,x_d,f):
        x_f = f(x)
        x = x+x_f
        x_d = x_d+x_f
        x_a = self.norm(x)
        x_d = self.norm(x_d)
        y = x_a+x_d
        return y,x_d