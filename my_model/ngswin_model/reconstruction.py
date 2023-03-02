import torch.nn as nn
import math

class Reconstruction(nn.Sequential):
    def __init__(self, target_mode, dim, out_chans=3):
        super(Reconstruction, self).__init__()
        
        self.upscale = upscale = int(target_mode[-1])
        self.target_mode = target_mode
        self.dim = dim
        self.out_chans = out_chans
            
        if target_mode in ['light_x2', 'light_x3', 'light_x4']:
            self.add_module('before_shuffle', nn.Conv2d(dim, out_chans*(upscale**2), 3, 1, 1))
            self.add_module('shuffler', nn.PixelShuffle(upscale)) # [B, dim/(upscale^2), upscale*H, upscale*W)                
            self.add_module('to_origin', nn.Conv2d(out_chans, out_chans, 3, 1, 1))
            
    def flops(self, resolutions):
        H, W = resolutions
        flops = 0
        if self.target_mode in ['light_x2', 'light_x3', 'light_x4']:
            # before shuffle: conv.weight, conv.bias
            flops += H*W * 3*3 * self.dim * self.out_chans*(self.upscale**2) + H*W * self.out_chans*(self.upscale**2)
            # to_origin: conv.weight, conv.bias
            flops += (self.upscale*H)*(self.upscale*W) * 3*3 * self.out_chans * self.out_chans + (self.upscale*H)*(self.upscale*W) * self.out_chans
        return flops