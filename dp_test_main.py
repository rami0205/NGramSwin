import argparse
from utils.argfunc_utils import *
from dp_builder import *
import os
from dp_engine_test import test_one_epoch
import torch.nn as nn

from my_model.ngswin_model.ngswin import NGswin
from my_model.swinirng import SwinIRNG

def get_args_parser():
    parser = argparse.ArgumentParser('N-Gram in Swin Transformers for Efficient Lightweight Image Super-Resolution', description='N-Gram in Swin Transformers for Efficient Lightweight Image Super-Resolution', add_help=True)
    # dp
    parser.add_argument('--device', type=str, default='cuda:0', help='first cuda device number')
    parser.add_argument('--num_device', type=int, default=4, help='number of gpus')
    # etc
    parser.add_argument('--model_time', type=str, help='automatically set when build model or manually set when load_model is True')
    parser.add_argument('--task', type=str, default='lightweight_sr')
    parser.add_argument('--load_model', type=str2bool, default=False, help='use checkpoint epoch or not')
    parser.add_argument('--checkpoint_epoch', type=int, default=0, help='restart train checkpoint')
    # model type
    parser.add_argument('--model_name', type=str, default='NGswin', help='NGswin or SwinIR-NG')
    parser.add_argument('--target_mode', type=str, default='light_x2', help='light_x2 light_x3 light_x4')
    parser.add_argument('--scale', type=int, help="upscale factor corresponding to 'target_mode'. it is automatically set.")
    parser.add_argument('--window_size', type=int, default=8, help='window size of (shifted) window attention')
    parser.add_argument('--training_patch_size', type=int, default=64, help='LQ image patch size. model input patch size only for training')
    # model ngram & in-channels spec -> ignored for SwinIR-NG
    parser.add_argument('--ngrams', type=str2tuple, default=(2,2,2,2), help='ngram size around each patch or window. embed-enc1-enc2-enc3-enc4-dec order.')
    parser.add_argument('--in_chans', type=int, default=3, help='number of input image channels')
    # model encoder spec -> ignored for SwinIR-NG
    parser.add_argument('--embed_dim', type=int, default=64, help='base dimension of model encoder')
    parser.add_argument('--depths', type=str2tuple, default=(6,4,4), help='number of transformer blocks on encoder')
    parser.add_argument('--num_heads', type=str2tuple, default=(6,4,4), help='number of multi-heads of self attention on encoder')
    parser.add_argument('--head_dim', type=int, help='dimension per multi-heads on encoder')
    # model decoder spec -> ignored for SwinIR-NG
    parser.add_argument('--dec_dim', type=int, default=64, help='base dimension of model decoder')
    parser.add_argument('--dec_depths', type=int, default=6, help='number of transformer blocks on decoder')
    parser.add_argument('--dec_num_heads', type=int, default=6, help='number of multi-heads of self attention on decoder')
    parser.add_argument('--dec_head_dim', type=int, help='each dimension per multi-heads on decoder')
    # model etc param spec -> ignored for SwinIR-NG
    parser.add_argument('--mlp_ratio', type=float, default=2.0, help="FFN's hidden dimension ratio over transformer dimension on encoder and decoder both")
    parser.add_argument('--qkv_bias', type=str2bool, default=True, help='whether using self attention qkv parameter bias on encoder and decoder both')
    # model dropout spec -> ignored for SwinIR-NG
    parser.add_argument('--drop_rate', type=float, default=0.0, help="dropout rate except attention layers")
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, help="dropout rate in attention layers")
    parser.add_argument('--drop_path_rate', type=float, default=0.0, help="stochastic drop rate of attention and ffn layers in transformer on encoder and decoder both")
    # model activation / norm / position-embedding spec -> ignored for SwinIR-NG
    parser.add_argument('--act_layer', type=str2nn_module, default=nn.GELU, help="activation layer")
    parser.add_argument('--norm_layer', type=str2nn_module, default=nn.LayerNorm, help="normalization layer")
    # train / test spec
    parser.add_argument('--test_only', type=str2bool, default=False, help='only evaluate model. not train')
    parser.add_argument('--sr_image_save', type=str2bool, default=True, help='save reconstructed image at test')
    # dataset / dataloader spec
    parser.add_argument('--img_norm', type=str2bool, default=True, help="image normalization before input") # -> ignored for SwinIR-NG
    
    return parser

def main(args):
    assert args.model_name in ['NGswin', 'SwinIR-NG'], "'model_name' should be NGswin or SwinIR-NG"
    if args.model_name == 'NGswin':
        model = NGswin(training_img_size=args.training_patch_size,
                       ngrams=args.ngrams,
                       in_chans=args.in_chans,
                       embed_dim=args.embed_dim,
                       depths=args.depths,
                       num_heads=args.num_heads,
                       head_dim=args.head_dim,
                       dec_dim=args.dec_dim,
                       dec_depths=args.dec_depths,
                       dec_num_heads=args.dec_num_heads,
                       dec_head_dim=args.dec_head_dim,
                       target_mode=args.target_mode,
                       window_size=args.window_size,
                       mlp_ratio=args.mlp_ratio,
                       qkv_bias=args.qkv_bias,
                       img_norm=args.img_norm,
                       drop_rate=args.drop_rate,
                       attn_drop_rate=args.attn_drop_rate,
                       drop_path_rate=args.drop_path_rate,
                       act_layer=args.act_layer,
                       norm_layer=args.norm_layer)
    else:
        model = SwinIRNG(upscale=args.scale, img_size=args.training_patch_size)
        
    print_complextity(model, args)
    
    sd = torch.load(f'pretrain/{args.model_name}_x{args.scale}.pth', map_location='cpu')
    missings,_ = model.load_state_dict(sd, strict=False)
    for xx in missings:
        assert 'relative_position_index' in xx or 'attn_mask' in xx, f'essential key {xx} is dropped!'
    print('<All keys matched successfully>')
        
    model = model.to(args.device)
    model = nn.DataParallel(model, device_ids=[x+int(args.device[-1]) for x in range(args.num_device)])
    
    args = record_args_after_build(args)
    test_one_epoch(model, 0, args)
            
if __name__ == '__main__':
    os.makedirs(f'./args/', exist_ok=True)
    os.makedirs(f'./logs/', exist_ok=True)
    parser = get_args_parser()
    args = parser.parse_args()
    
    args.scale = int(args.target_mode[-1]) if args.target_mode[-1].isdigit() else 1
    
    args = record_args_before_build(args)
    main(args)