# strongly following Swin Transformer code
# Written by Haram Choi

import torch
import torch.nn as nn
from timm.models.layers import _assert, to_2tuple, DropPath, Mlp

from .win_partition import NGramWindowPartition, window_partition, window_unpartition
from .win_attention import WindowAttention

class NSTB(nn.Module): # N-Gram Swin Transformer Block
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        training_num_patches: Number of height and width of the patch only when training.
        ngram (int): ngram: (int): How much windows to see as context.
        num_heads (int): Number of attention heads.
        window_size (int): The size of the window.
        shift_size (int): Shift size for SW-MSA.
        head_dim (int, optional): Number of channels per head (dim // num_heads if not set).  Default: None
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    Inputs:
        [B, ph*pw, D], (ph, pw)
    Returns:
        [B, ph*pw, D], (ph, pw)
    """

    def __init__(
        self, dim, training_num_patches, ngram, num_heads, window_size, shift_size,
        head_dim=None, mlp_ratio=2., qkv_bias=True, 
        drop=0., attn_drop=0., drop_path=0.,
        act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        
        super().__init__()
        self.dim = dim
        self.training_num_patches = training_num_patches
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        _assert(0 <= self.shift_size < self.window_size, "shift_size must in 0~window_size")
        
        self.ngram_window_partition = NGramWindowPartition(dim, window_size, ngram, num_heads, shift_size=shift_size)
        self.attn = WindowAttention(
            dim, num_heads=num_heads, head_dim=head_dim, window_size=to_2tuple(self.window_size),
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        # window mask
        attn_mask = self.make_mask(training_num_patches) if shift_size>0 else None
        self.register_buffer("attn_mask", attn_mask)
        
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ffn = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        
    def make_mask(self, num_patches):
        ph, pw = num_patches
        img_mask = torch.zeros((1, ph, pw, 1)) # [1, ph, pw, 1]
        cnt = 0
        for h in (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)):
            for w in (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None)):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows, (wh,ww) = window_partition(img_mask, self.window_size)  # [wh*ww*1, WH, WW, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # [wh*ww, WH*WW]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # [wh*ww, WH, WW]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0)) # [wh*ww, WH, WW]
        return attn_mask
    
    def _attention(self, x, num_patches):
        # window partition - (cyclic shift) - cosine attention - window unpartition - (reverse shift)
        ph, pw = num_patches
        B, p, D = x.size()
        _assert(p == ph * pw, f"size is wrong!")
        
        x = x.view(B, ph, pw, D) # [B, ph, pw, D], Unembedding
        
        # N-Gram Window Partition (-> cyclic shift)
        x_windows, (wh,ww) = self.ngram_window_partition(x) # [B*wh*ww, WH, WW, D], (wh, ww)
        
        x_windows = x_windows.view(-1, self.window_size * self.window_size, D)  # [B*wh*ww, WH*WW, D], Re-embedding
        
        # W-MSA/SW-MSA
        if self.training_num_patches==num_patches:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # [B*wh*ww, WH*WW, D]
        else:
            eval_attn_mask = self.make_mask(num_patches).to(x.device) if self.shift_size>0 else None
            attn_windows = self.attn(x_windows, mask=eval_attn_mask)  # [B*wh*ww, WH*WW, D]
        
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, D) # [B*wh*ww, WH, WW, D], Unembedding
        
        # Window Unpartition
        shifted_x = window_unpartition(attn_windows, (wh,ww))  # [B, ph, pw, D]
        
        # Reverse Cyclic Shift
        reversed_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) if self.shift_size > 0 else shifted_x # [B, ph, pw, D]
        reversed_x = reversed_x.view(B, ph*pw, D) # [B, ph*pw, D], Re-embedding
        
        return reversed_x

    def forward(self, x, num_patches):
        x_ = x
        # (S)W Attention -> Layer-Norm -> Drop-Path -> Skip-Connection
        x = x + self.drop_path(self.norm1(self._attention(x, num_patches))) # [B, ph*pw, D]
        # FFN -> Layer-Norm -> Drop-Path -> Skip-Connection
        x = x + self.drop_path(self.norm2(self.ffn(x))) # [B, ph*pw, 4D] -> [B, ph*pw, D]
        return x_, x, num_patches

    def flops(self, resolutions):
        H, W = resolutions
        flops = 0
        # ngram_window_partition
        flops += self.ngram_window_partition.flops((H,W))
        # (S)W-MSA
        num_windows = H//self.window_size * W//self.window_size
        flops += self.attn.flops(num_windows)
        # norm1
        flops += H*W * self.dim
        # FFN
        flops += H*W * self.dim * self.mlp_ratio*self.dim + self.mlp_ratio*self.dim # fc1: linear.weight, linear.bias
        flops += H*W * self.mlp_ratio*self.dim * self.dim + self.dim # fc2: linear.weight, linear.bias
        flops = int(flops)
        # norm2
        flops += H*W * self.dim
        return flops