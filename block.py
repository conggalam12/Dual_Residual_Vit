# Transformer Building block implementation
import torch.nn as nn
from attention import AttentionModel
from mlp import MLP
from Dual_Residual import Dual_Residual_Block
class BuildingBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio = 4.0, include_bias = True, dropout_p = 0.5, attention_p = 0.5):
        super(BuildingBlock, self).__init__()
        self.norm= nn.LayerNorm(dim, eps=1e-6)
        self.attention = AttentionModel(dim, num_heads, include_bias, attention_p, dropout_p)
        self.hidden_features = int(dim * mlp_ratio)
        self.FFN = nn.Sequential(
            nn.Linear(dim, dim*3),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(dim*3,dim),
            nn.Dropout(dropout_p)
        )
        self.mlp = MLP(dim, self.hidden_features, dim)
        self.dim = dim
        self.Residual = Dual_Residual_Block(dim,self.norm)
    def forward(self, x):
        #Block 1
        x_ln1,x_d = self.Residual(x,x,self.attention)
        x_ln2,x_d = self.Residual(x_ln1,x_d,self.FFN)

        #Block 2
        x_ln3,x_d = self.Residual(x_ln2,x_d,self.attention)
        x_ln4,x_d = self.Residual(x_ln3,x_d,self.FFN)

        #Block 3
        x_ln5,x_d = self.Residual(x_ln4,x_d,self.attention)
        x_ln6,x_d = self.Residual(x_ln5,x_d,self.FFN)

        #Block 4:
        x_ln7,x_d = self.Residual(x_ln6,x_d,self.attention)
        x_ln8,x_d = self.Residual(x_ln7,x_d,self.FFN)
        
        y = self.mlp(x_ln8)
        return y
        