# Written by Haram Choi
import torch
import torch.nn as nn
from timm.models.layers import _assert
from einops import rearrange
from einops.layers.torch import Rearrange

from .pool import BottleneckPool
from .nstb import NSTB

class ShallowExtractor(nn.Module):
    """ Shallow Module.

    Args:
        in_chans (int): Number of input image channels.
        out_chans (int): Number of output feature map channels.
    """
    def __init__(self, in_chans, out_chans):
        super(ShallowExtractor, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        
        self.conv1 = nn.Conv2d(in_chans, out_chans, 3, 1, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        return x
    
    def flops(self, resolutions):
        H, W = resolutions
        flops = 0
        flops += H*W * 3*3 * self.in_chans * self.out_chans + H*W * self.out_chans # conv1: conv.weight, conv.bias
        return flops

class PatchMerging(nn.Module):
    """ Patch Merging Layer.

    Args:
        dim (int): Number of input dimension (channels).
        downsample_dim (int, optional): Number of output dimension (channels) (dim if not set).  Default: None
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, downsample_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.downsample_dim = downsample_dim or dim
        self.norm = norm_layer(4*dim)
        self.reduction = nn.Linear(4*dim, self.downsample_dim, bias=False)

    def forward(self, x, num_patches):
        """
        x: B, ph*pw, C
        """
        ph, pw = num_patches
        B, p, D = x.size()
        _assert(p == ph * pw, "size is wrong!")
        _assert(ph % 2 == 0 and pw % 2 == 0, f"number of patches ({ph}*{pw}) is not even!")

        x = x.view(B, ph, pw, D) # [B, ph, pw, D], Unembedding
        
        # Concat 2x2
        x0 = x[:, 0::2, 0::2, :]  # [B, ph/2, pw/2, D], top-left
        x1 = x[:, 0::2, 1::2, :]  # [B, ph/2, pw/2, D], top-right
        x2 = x[:, 1::2, 0::2, :]  # [B, ph/2, pw/2, D], bottom-left
        x3 = x[:, 1::2, 1::2, :]  # [B, ph/2, pw/2, D], bottom-right
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, ph/2, pw/2, 4D]
        _, new_ph, new_pw, _ = x.size()
        x = x.view(B, -1, 4*D)  # [B, (ph*pw)/4, 4D]
        
        x = self.norm(x)
        x = self.reduction(x) # [B, (ph*pw)/4, D]

        return x, (new_ph, new_pw)
    
    def flops(self, resolutions):
        H, W = resolutions
        flops = 0
        # norm
        flops = H*W * 4*self.dim
        # reduction
        flops += (H//2)*(W//2) * 4*self.dim * self.downsample_dim + self.downsample_dim # linear.weight, linear.bias
        return flops

class EncoderLayer(nn.Module):
    """ A N-Gram Context Swin Transformer Encoder Layer for one stage.

    Args:
        dim (int): Number of input dimension (channels).
        training_num_patches (tuple[int]): Number of height and width of the patch only when training.
        ngram (int): How much windows to see as context.
        depth (int): Number of NSTBs.
        num_heads (int): Number of attention heads.
        window_size (int): The size of the window.
        head_dim (int, optional): Channels per head (dim // num_heads if not set).  Default: None
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim.  Default: 2.0
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value.  Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        downsample_dim (int, optional): Number of output channels (dim if not set).  Default: None
        num_cas (int, optional): Number of accumulative concatenation.  Default: 1
    """
    
    def __init__(
        self, dim, training_num_patches, ngram, depth, num_heads, window_size,
        head_dim=None, mlp_ratio=2., qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0,
        act_layer=nn.GELU, norm_layer=nn.LayerNorm, downsample=None, downsample_dim=None, num_cas=1):
        
        super(EncoderLayer, self).__init__()
        self.dim = dim
        self.num_cas = num_cas
        
        self.across_cascade_proj = nn.Linear(dim*num_cas, dim) if num_cas!=1 else nn.Identity()
        
        # build blocks
        self.blocks = nn.Sequential(*[
            NSTB(
                dim=dim, training_num_patches=training_num_patches, ngram=ngram, num_heads=num_heads,
                window_size=window_size, shift_size=0 if (i%2==0) else window_size//2,
                head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, 
                act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth)])
        
        # patch-merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, downsample_dim=downsample_dim, norm_layer=norm_layer)
        else:
            self.downsample = None
            
    def forward(self, x, num_patches):
        x = self.across_cascade_proj(x)
        
        x_ = 0 # for within stage residual connection including patch-merging
        for i, blk in enumerate(self.blocks):
            # Within Stage Residual Connection including patch-merging
            x_, x, num_patches = blk(x+x_, num_patches)
            
        x_down, num_patches = self.downsample(x+x_, num_patches) if self.downsample is not None else (x, num_patches)
        
        return x, x_down, num_patches
    
    def flops(self, resolutions):
        H, W = resolutions
        flops = 0
        # across_cascade_proj: linear.weight, linear.bias
        flops += H*W * self.num_cas*self.dim * self.dim + self.dim if isinstance(self.across_cascade_proj, nn.Linear) else 0
        # NSTB blocks: ngram_window_partition, (S)W-MSA, norm1, FFN, norm2
        for blk in self.blocks:
            flops += blk.flops((H,W))
        # patch merging
        flops += self.downsample.flops((H,W)) if self.downsample is not None else 0
        return flops
    
def pixel_shuffle_permute(x, num_patches, out_size):
    scale_h = out_size[0]//num_patches[0]
    scale_w = out_size[1]//num_patches[1]
    
    x = rearrange(x, 'b (h w) (c ch cw) -> b (h ch w cw) c', h=num_patches[0], ch=scale_h, cw=scale_w)
    return x

class SCDPBottleneck(nn.Module):
    """pixel-Shuffle, Concatenate, Depth-wise conv, and Point-wise project Bottleneck block
    Args:
        num_encoder_stages (int): Number of encoder NSTB stages.
        enc_dim (int): Number of encoder dimension (channels).
        dec_dim (int): Number of decoder dimension (channels).
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """
    
    def __init__(self, num_encoder_stages, enc_dim, dec_dim, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(SCDPBottleneck, self).__init__()
        self.num_encoder_stages = num_encoder_stages
        self.dec_dim = dec_dim
        
        self.bottleneck_pool = BottleneckPool(enc_dim)
        self.concat_dim = sum([4**x for x in range(num_encoder_stages)])*(enc_dim//16)
        self.depthwise = nn.Conv2d(self.concat_dim, self.concat_dim, 3, 1, 1, groups=self.concat_dim)
        self.act = act_layer()
        self.pointwise = nn.Linear(self.concat_dim, dec_dim)
        self.norm = norm_layer(dec_dim)
        
    def forward(self, shallow, x_list, num_patches_list):
        _assert(len(x_list)==self.num_encoder_stages, "number of elements in input list is wrong")
        
        x_list = [pixel_shuffle_permute(x+self.bottleneck_pool(shallow, i), num_patches_list[i], num_patches_list[0]) \
                  for i, x in enumerate(x_list)] # "S" pixel-Shuffle. upsample each input
        x = torch.cat(x_list, dim=-1) # "C" Concatenate
        x = rearrange(x, 'b (h w) c -> b c h w', h=num_patches_list[0][0]) # unembedding
        x = self.act(self.depthwise(x)) # "D" Depth-wise conv
        x = rearrange(x, 'b c h w -> b (h w) c') # re-embedding
        x = self.pointwise(x) # "P" Point-wise project
        x = self.norm(x)
        
        return x, num_patches_list[0]
    
    def flops(self, resolutions):
        H, W = resolutions
        flops = 0
        # pixel-Shuffle
        for i in range(3):
            Hr, Wr = H//(2**i), W//(2**i)
            flops += self.bottleneck_pool.flops((H,W), (Hr,Wr))
        # Depth-wise
        flops += H*W * 3*3 * self.concat_dim + H*W * self.concat_dim + H*W*self.concat_dim # conv.weight, conv.bias, act
        # Point-wise
        flops += H*W * self.concat_dim * self.dec_dim + self.dec_dim # linear.weight, linear.bias
        return flops

class DecoderLayer(nn.Module):
    """ A N-Gram Context Swin Transformer Decoder Layer for one stage.
    Args:
        dim (int): Number of input dimension (channels).
        training_num_patches (tuple[int]): Number of height and width of the patch only when training.
        ngram (int): ngram: (int): How much windows to see as context.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): The size of the window.
        head_dim (int, optional): Channels per head (dim // num_heads if not set).  Default: None
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim.  Default: 2.0
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        last (bool, optional): The final layer of decoder stages or not.  Default: False
    """
    
    def __init__(
        self, dim, training_num_patches, ngram, depth, num_heads,
        window_size, head_dim=None, mlp_ratio=2., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
        act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        
        super(DecoderLayer, self).__init__()
        self.dim = dim
        
        self.blocks = nn.Sequential(*[
            NSTB(
                dim=dim, training_num_patches=training_num_patches, ngram=ngram,
                num_heads=num_heads, window_size=window_size, shift_size=0 if (i%2==0) else window_size//2, 
                head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, 
                act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth)])
    
    def forward(self, x, num_patches, out_size=None):
        x_ = 0 # for within stage residual connection
        for i, blk in enumerate(self.blocks):
            # Within Stage Residual Connection
            x_, x, num_patches = blk(x+x_, num_patches)
            
        return x, out_size
    
    def flops(self, resolutions):
        H, W = resolutions
        flops = 0
        # NSTB blocks: ngram_window_partition, (S)W-MSA, norm1, FFN, norm2
        for blk in self.blocks:
            flops += blk.flops((H,W))
        return flops