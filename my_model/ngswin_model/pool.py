# Written by Haram Choi
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

class InterPool(nn.Module):
    def __init__(self, dim):
        super(InterPool, self).__init__()
        self.dim = dim
        self.max_pool = nn.MaxPool2d(2)
        
    def forward(self, x, num_patches):
        x = rearrange(x, 'b (h w) c -> b c h w', h=num_patches[0])
        x = self.max_pool(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x
    
    def flops(self, resolutions):
        H, W, D = resolutions
        flops = 0
        flops += (H//2)*(W//2) * 2*2 * D # max_pool
        return flops
    
class BottleneckPool(nn.Module):
    def __init__(self, dim):
        super(BottleneckPool, self).__init__()
        self.dim = dim
        
        self.max_pool = nn.MaxPool2d(2)
        self.act = nn.LeakyReLU()
    def forward(self, x, exp):
        for _ in range(exp):
            x = self.max_pool(x)
        x = self.act(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x
    
    def flops(self, resolutions_origin, resolutions):
        import math
        Ho, Wo, H, W = resolutions_origin+resolutions
        exp = int(math.log2(Ho//H))
        flops = 0
        Hr, Wr = Ho//2, Wo//2
        for e in range(exp):
            flops += Hr*Wr * 2*2 * self.dim # max_pool
            Hr, Wr = Hr//2, Wr//2
        flops += H*W*self.dim # act
        return flops