# Proposed N-Gram Context method is in this code.
# except that, strongly following Swin Transformer code
# Written by Haram Choi
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import _assert, to_2tuple
from timm.models.fx_features import register_notrace_function
from torchvision.transforms import functional as TF

from .win_attention import WindowAttention
    
class NGramContext(nn.Module):
    '''
    Args:
        dim (int): Number of input channels.
        window_size (int or tuple[int]): The height and width of the window.
        ngram (int): How much windows(or patches) to see.
        ngram_num_heads (int):
        padding_mode (str, optional): How to pad.  Default: seq_refl_win_pad
                                                   Options: ['seq_refl_win_pad', 'zero_pad']
    Inputs:
        x: [B, ph, pw D] or [B, C, H, W]
    Returns:
        context: [B, wh, ww, 1, 1, D] or [B, C, ph, pw]
    '''
    def __init__(self, dim, window_size, ngram, ngram_num_heads, padding_mode='seq_refl_win_pad'):
        super(NGramContext, self).__init__()
        _assert(padding_mode in ['seq_refl_win_pad', 'zero_pad'], "padding mode should be 'seq_refl_win_pad' or 'zero_pad'!!")
        
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.ngram = ngram
        self.padding_mode = padding_mode
        
        self.unigram_embed = nn.Conv2d(dim, dim//2,
                                       kernel_size=(self.window_size[0], self.window_size[1]), 
                                       stride=self.window_size, padding=0, groups=dim//2)
        self.ngram_attn = WindowAttention(dim=dim//2, num_heads=ngram_num_heads, window_size=ngram)
        self.avg_pool = nn.AvgPool2d(ngram)
        self.merge = nn.Conv2d(dim, dim, 1, 1, 0)
        
    def seq_refl_win_pad(self, x, back=False):
        if self.ngram == 1: return x
        x = TF.pad(x, (0,0,self.ngram-1,self.ngram-1)) if not back else TF.pad(x, (self.ngram-1,self.ngram-1,0,0))
        if self.padding_mode == 'zero_pad':
            return x
        if not back:
            (start_h, start_w), (end_h, end_w) = to_2tuple(-2*self.ngram+1), to_2tuple(-self.ngram)
            # pad lower
            x[:,:,-(self.ngram-1):,:] = x[:,:,start_h:end_h,:]
            # pad right
            x[:,:,:,-(self.ngram-1):] = x[:,:,:,start_w:end_w]
        else:
            (start_h, start_w), (end_h, end_w) = to_2tuple(self.ngram), to_2tuple(2*self.ngram-1)
            # pad upper
            x[:,:,:self.ngram-1,:] = x[:,:,start_h:end_h,:]
            # pad left
            x[:,:,:,:self.ngram-1] = x[:,:,:,start_w:end_w]
            
        return x
    
    def sliding_window_attention(self, unigram):
        slide = unigram.unfold(3, self.ngram, 1).unfold(2, self.ngram, 1)
        slide = rearrange(slide, 'b c h w ww hh -> b (h hh) (w ww) c') # [B, 2(wh+ngram-2), 2(ww+ngram-2), D/2]
        slide, num_windows = window_partition(slide, self.ngram) # [B*wh*ww, ngram, ngram, D/2], (wh, ww)
        slide = slide.view(-1, self.ngram*self.ngram, self.dim//2) # [B*wh*ww, ngram*ngram, D/2]
        
        context = self.ngram_attn(slide).view(-1, self.ngram, self.ngram, self.dim//2) # [B*wh*ww, ngram, ngram, D/2]
        context = window_unpartition(context, num_windows) # [B, wh*ngram, ww*ngram, D/2]
        context = rearrange(context, 'b h w d -> b d h w') # [B, D/2, wh*ngram, ww*ngram]
        context = self.avg_pool(context) # [B, D/2, wh, ww]
        return context
        
    def forward(self, x):
        B, ph, pw, D = x.size()
        x = rearrange(x, 'b ph pw d -> b d ph pw') # [B, D, ph, pw]
        unigram = self.unigram_embed(x) # [B, D/2, wh, ww]
        
        unigram_forward_pad = self.seq_refl_win_pad(unigram, False) # [B, D/2, wh+ngram-1, ww+ngram-1]
        unigram_backward_pad = self.seq_refl_win_pad(unigram, True) # [B, D/2, wh+ngram-1, ww+ngram-1]
        
        context_forward = self.sliding_window_attention(unigram_forward_pad) # [B, D/2, wh, ww]
        context_backward = self.sliding_window_attention(unigram_backward_pad) # [B, D/2, wh, ww]
        
        context_bidirect = torch.cat([context_forward, context_backward], dim=1) # [B, D, wh, ww]
        context_bidirect = self.merge(context_bidirect) # [B, D, wh, ww]
        context_bidirect = rearrange(context_bidirect, 'b d h w -> b h w d') # [B, wh, ww, D]
        
        return context_bidirect.unsqueeze(-2).unsqueeze(-2).contiguous() # [B, wh, ww, 1, 1, D]
    
    def flops(self, resolutions):
        H, W = resolutions
        wh, ww = H//self.window_size[0], W//self.window_size[1]
        flops = 0
        # unigram embed: conv.weight, conv.bias
        flops += wh*ww * self.window_size[0]*self.window_size[1] * self.dim + wh*ww * self.dim
        # ngram sliding attention (forward & backward)
        flops += 2*self.ngram_attn.flops(wh*ww)
        # avg pool
        flops += wh*ww * 2*2 * self.dim
        # merge concat
        flops += wh*ww * 1*1 * self.dim * self.dim
        return flops

class NGramWindowPartition(nn.Module):
    """
    Args:
        dim (int): Number of input channels.
        window_size (int): The height and width of the window.
        ngram (int): How much windows to see as context.
        ngram_num_heads (int):
        shift_size (int, optional): Shift size for SW-MSA.  Default: 0
    Inputs:
        x: [B, ph, pw, D]
    Returns:
        [B*wh*ww, WH, WW, D], (wh, ww)
    """
    def __init__(self, dim, window_size, ngram, ngram_num_heads, shift_size=0):
        super(NGramWindowPartition, self).__init__()
        self.window_size = window_size
        self.ngram = ngram
        self.shift_size = shift_size
        
        self.ngram_context = NGramContext(dim, window_size, ngram, ngram_num_heads, padding_mode='seq_refl_win_pad')
    
    def forward(self, x):
        B, ph, pw, D = x.size()
        wh, ww = ph//self.window_size, pw//self.window_size # number of windows (height, width)
        _assert(0 not in [wh, ww], "feature map size should be larger than window size!")
        
        context = self.ngram_context(x) # [B, wh, ww, 1, 1, D]
        
        windows = rearrange(x, 'b (h wh) (w ww) c -> b h w wh ww c', 
                            wh=self.window_size, ww=self.window_size).contiguous() # [B, wh, ww, WH, WW, D]. semi window partitioning
        windows+=context # [B, wh, ww, WH, WW, D]. inject context
        
        # Cyclic Shift
        if self.shift_size>0:
            x = rearrange(windows, 'b h w wh ww c -> b (h wh) (w ww) c').contiguous() # [B, ph, pw, D]. re-patchfying
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) # [B, ph, pw, D]. cyclic shift
            windows = rearrange(shifted_x, 'b (h wh) (w ww) c -> b h w wh ww c', 
                                wh=self.window_size, ww=self.window_size).contiguous() # [B*wh*ww, WH, WW, D]. re-semi window partitioning
            
        windows = rearrange(windows, 'b h w wh ww c -> (b h w) wh ww c').contiguous() # [B*wh*ww, WH, WW, D]. window partitioning
        
        return windows, (wh, ww)
    
    def flops(self, resolutions):
        H, W = resolutions
        flops = 0
        flops += self.ngram_context.flops((H,W))
        return flops

def window_partition(x, window_size: int):
    """
    Args:
        x: [B, ph, pw, D]
        window_size (int): The height and width of the window.
    Returns:
        [B*wh*ww, WH, WW, D], (wh, ww)
    """
    B, ph, pw, D = x.size()
    wh, ww = ph//window_size, pw//window_size # number of windows (height, width)
    if 0 in [wh, ww]:
        # if feature map size is smaller than window size, do not partition
        return x, (wh, ww)
    windows = rearrange(x, 'b (h wh) (w ww) c -> (b h w) wh ww c', wh=window_size, ww=window_size).contiguous()
    return windows.contiguous(), (wh, ww)
    
@register_notrace_function  # reason: int argument is a Proxy
def window_unpartition(windows, num_windows):
    """
    Args:
        windows: [B*wh*ww, WH, WW, D]
        num_windows (tuple[int]): The height and width of the window.
    Returns:
        x: [B, ph, pw, D]
    """
    x = rearrange(windows, '(p h w) wh ww c -> p (h wh) (w ww) c', h=num_windows[0], w=num_windows[1])
    return x.contiguous()