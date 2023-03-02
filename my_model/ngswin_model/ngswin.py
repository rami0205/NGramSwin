# Written by Haram Choi
import math
import torch
import torch.nn as nn
from einops import rearrange

from torchvision.transforms import functional as TF
from timm.models.layers import to_2tuple, to_ntuple, _assert, trunc_normal_
from timm.models.helpers import named_apply
from timm.models.vision_transformer import get_init_weights_vit
from timm.models.efficientnet_builder import _init_weight_goog

from .pool import InterPool
from .main_branch import ShallowExtractor, PatchMerging, EncoderLayer, SCDPBottleneck, DecoderLayer
from .reconstruction import Reconstruction
from utils.etc_utils import denormalize

class NGswin(nn.Module):
    """
    Args:
        training_img_size (int or tuple[int]): Input image size.  Default 64
        ngrams tuple[int]: How much windows to see as context in each encoder and decoder.  Default: (2,2,2,2)
        in_chans (int): Number of input image channels.  Default: 3
        embed_dim (int): Patch embedding dimension. Dimension of all encoder layers.  Default: 64
        depths (tuple[int]): Depth of each encoder stage. i.e., number of NSTBs.  Default: (6,4,4)
        num_heads (tuple[int]): Number of attention heads in each encoder layer.  Default: (6,4,4)
        head_dim (int or tuple[int]): Channels per head of each encoder layer (dim // num_heads if not set).  Default: None
        dec_dim (int): Dimension of decoder stage.  Default: 64
        dec_depths (int): Depth of a decoder stage.  Default: 6
        dec_num_heads (int): Number of attention heads in a decoder layer.  Default: 6
        dec_head_dim (int): Channels per head of decoder layer (dim // num_heads if not set).  Default: None
        target_mode (str): light_x2, light_x3, light_x4.  Default: light_x2
        window_size (int): The size of the window.  Default: 8
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.  Default: 2.0
        qkv_bias (bool): If True, add a learnable bias to query, key, value.  Default: True
        drop_rate (float): Dropout rate.  Default: 0.0
        attn_drop_rate (float): Attention dropout rate.  Default: 0.0
        drop_path_rate (float): Stochastic depth rate.  Default: 0.0
        act_layer (nn.Module, optional): Activation layer.  Default: nn.GELU
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        weight_init (str): The way to initialize weights of layers.  Default: ''
    """
    def __init__(
        self, training_img_size=64, ngrams=(2,2,2,2),
        in_chans=3, embed_dim=64, depths=(6,4,4), num_heads=(6,4,4), head_dim=None,
        dec_dim=64, dec_depths=6, dec_num_heads=6, dec_head_dim=None,
        target_mode='light_x2', window_size=8, mlp_ratio=2., qkv_bias=True, img_norm=True,
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
        act_layer=nn.GELU, norm_layer=nn.LayerNorm, weight_init='', **kwargs):
        
        _assert(target_mode in ['light_x2', 'light_x3', 'light_x4'], 
                "'target mode' should be one of ['light_x2', 'light_x3', 'light_x4']")
        
        super(NGswin, self).__init__()
        self.training_img_size = to_2tuple(training_img_size)
        self.ngrams = ngrams
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        dec_depths = (dec_depths,)
        dec_num_heads = (dec_num_heads,)
        self.dec_dim=dec_dim
        self.dec_depths=dec_depths
        self.dec_num_heads=dec_num_heads
        self.window_size = window_size
        self.num_encoder_stages = len(depths)
        self.num_decoder_stages = len(dec_depths)
        self.scale = int(target_mode[-1])
        
        if self.scale==2:
            self.mean = (0.4485, 0.4375, 0.4045)
            self.std = (0.2397, 0.2290, 0.2389)
        elif self.scale==3:
            self.mean = (0.4485, 0.4375, 0.4045)
            self.std = (0.2373, 0.2265, 0.2367)
        elif self.scale==4:
            self.mean = (0.4485, 0.4375, 0.4045)
            self.std = (0.2352, 0.2244, 0.2349)
        self.img_norm = img_norm
        
        head_dim = to_ntuple(self.num_encoder_stages)(head_dim) # default None
        dec_head_dim = to_ntuple(self.num_decoder_stages)(dec_head_dim) # default None
        mlp_ratio = to_ntuple(self.num_encoder_stages+self.num_decoder_stages)(mlp_ratio)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths)+sum(dec_depths))]  # stochastic depth decay rule
        
        # Main Branch
        self.shallow_extract = ShallowExtractor(in_chans, embed_dim)
        self.inter_pool = InterPool(embed_dim)
        for i in range(self.num_encoder_stages):
            self.add_module(f'encoder_layer{i+1}', EncoderLayer(
                                                    dim=embed_dim,
                                                    training_num_patches=(self.training_img_size[0]//2**i,
                                                                          self.training_img_size[1]//2**i),
                                                    ngram=ngrams[i],
                                                    depth=depths[i],
                                                    num_heads=num_heads[i],
                                                    window_size=window_size,
                                                    head_dim=head_dim[i],
                                                    mlp_ratio=mlp_ratio[i],
                                                    qkv_bias=qkv_bias,
                                                    drop=drop_rate,
                                                    attn_drop=attn_drop_rate,
                                                    drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                                    act_layer=act_layer,
                                                    norm_layer=norm_layer,
                                                    downsample=PatchMerging if (i+1)!=self.num_encoder_stages else None,
                                                    downsample_dim=embed_dim if (i+1)!=self.num_encoder_stages else None,
                                                    num_cas=i+1))
        self.bottleneck = SCDPBottleneck(num_encoder_stages=self.num_encoder_stages, 
                                         enc_dim=embed_dim, dec_dim=dec_dim, 
                                         act_layer=act_layer, norm_layer=norm_layer)
        for i in range(self.num_decoder_stages):
            self.add_module(f'decoder_layer{i+1}', DecoderLayer(
                                                    dim=dec_dim,
                                                    training_num_patches=(self.training_img_size[0]//2**(1-i),
                                                                          self.training_img_size[1]//2**(1-i)),
                                                    ngram=ngrams[self.num_encoder_stages+i],
                                                    depth=dec_depths[i],
                                                    num_heads=dec_num_heads[i],
                                                    window_size=window_size,
                                                    head_dim=dec_head_dim[i],
                                                    mlp_ratio=mlp_ratio[self.num_encoder_stages+i],
                                                    qkv_bias=qkv_bias,
                                                    drop=drop_rate,
                                                    attn_drop=attn_drop_rate,
                                                    drop_path=dpr[sum(depths)+sum(dec_depths[:i]):sum(depths)+sum(dec_depths[:i+1])],
                                                    act_layer=act_layer,
                                                    norm_layer=norm_layer))
        self.norm = norm_layer(dec_dim)
        
        # Reconstruction
        self.to_target = Reconstruction(target_mode, dec_dim, in_chans)
        
        self.apply(self._init_weights)
        
#         if weight_init != 'skip':
#             self.init_weights(weight_init)

            
    def _init_weights(self, m):
        # Swin V2 manner
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = set()
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd
    
    def forward_size_norm(self, x):
        _,_,h,w = x.size()
        unit = 4*self.window_size
        padh = unit-(h%unit) if h%unit!=0 else 0
        padw = unit-(w%unit) if w%unit!=0 else 0
        x = TF.pad(x, (0,0,padw,padh))
        return TF.normalize(x, self.mean, self.std) if self.img_norm else x

    def forward_encoder(self, x):
        _,_,H,W = x.size()
        x = self.shallow_extract(x)
        shallow_short = x
        
        # Across Stage Pooling Cascading
        c0, num_patches0 = rearrange(x, 'b c h w -> b (h w) c'), (H,W) # [B, HW, 1D], (H, W). ASPC
        e1_, e1, num_patches1 = self.encoder_layer1(c0, num_patches0) # [B, HW, D], [B, HW/(2^2), D], (H/2, W/2). encoder-stage1
        
        c1 = torch.cat([self.inter_pool(c0,num_patches0), e1], dim=-1) # [B, HW/(2^2), 2D], (H, W). ASPC
        e2_, e2, num_patches2 = self.encoder_layer2(c1, num_patches1) # [B, HW/(2^2), D], [B, HW/(4^2), D], (H/4, W/4). encoder-stage2
        
        c2 = torch.cat([self.inter_pool(c1,num_patches1), e2], dim=-1) # [B, HW/(4^2), 3D], (H, W). ASPC
        e3_, e3, num_patches3 = self.encoder_layer3(c2, num_patches2) # [B, HW/(4^2), D], [B, HW/(8^2), D], (H/8, W/8). encoder-stage3
        
        num_patches_list = [num_patches0, num_patches1, num_patches2] # (H, W), (H/2, W/2), (H/4, W/4)
        
        # bottleneck
        out, num_patches_scdp = self.bottleneck(shallow_short, [e1_,e2_,e3_], num_patches_list) # [B, HW, D], (H, W)
        
        return out, num_patches_scdp, c0, e1_
    
    def forward_decoder(self, x, num_patches, out_size, e1_):
        x, num_pathces = self.decoder_layer1(x+e1_, num_patches) # [B, HW, D], (H, W) enc-dec skip connection
        out = self.norm(x)
        return out
    
    def forward_reconstruct(self, x, img_size):
        x = rearrange(x, 'p (h w) c -> p c h w', h=img_size[0]) # for pixel-shffule and cnn
        out = self.to_target(x) # reconstruct to target image size
        return out
    
    def forward(self, x):
        _,_,H_ori,W_ori = x.size()
        x = self.forward_size_norm(x)
        B, C, H, W = x.size()
        x, num_patches, shallow, e1_ = self.forward_encoder(x)
        dec_out = self.forward_decoder(x, num_patches, (H,W), e1_)
        dec_out += shallow # global skip connection
        out = self.forward_reconstruct(dec_out, (H,W))
        
        out = denormalize(out, self.mean, self.std) if self.img_norm else out
        out = out[:, :, :H_ori*self.scale, :W_ori*self.scale]
        return out
    
    def flops(self, resolutions):
        unit = 4*self.window_size
        D = self.embed_dim
        H, W = resolutions
        H += unit-(H%unit) if H%unit!=0 else 0
        W += unit-(W%unit) if W%unit!=0 else 0
        flops = 0
        # shallow feature
        flops += self.shallow_extract.flops((H,W))
        # encoder1
        flops += self.encoder_layer1.flops((H,W))
        # encoder2
        flops += self.inter_pool.flops((H,W,D))
        flops += self.encoder_layer2.flops((H//2,W//2))
        # encoder3
        flops += self.inter_pool.flops((H//2,W//2,2*D))
        flops += self.encoder_layer3.flops((H//4,W//4))
        # bottleneck
        flops += self.bottleneck.flops((H,W))
        # decoder
        flops += self.decoder_layer1.flops((H,W))
        # norm
        flops += H*W * self.embed_dim
        # reconstruct
        flops += self.to_target.flops((H,W))
            
        return flops
