import argparse
from utils.argfunc_utils import *
from dp_builder import *
import os
from dp_engine_finetune import train_one_epoch
from dp_engine_test import test_one_epoch

def get_args_parser():
    parser = argparse.ArgumentParser('N-Gram in Swin Transformers for Efficient Lightweight Image Super-Resolution', description='N-Gram in Swin Transformers for Efficient Lightweight Image Super-Resolution', add_help=True)
    # dp
    parser.add_argument('--device', type=str, default='cuda:0', help='first cuda device number')
    parser.add_argument('--num_device', type=int, default=4, help='number of gpus')
    # pre-trained info
    parser.add_argument('--pretrain_path', type=str, help='pretrained model .pth file path')
    parser.add_argument('--warm_start', type=str2bool, default=True, help='do or not warm-start when fine-tuning')
    parser.add_argument('--warm_start_epoch', type=int, default=50, help='how much warm-start before whole fine-tuning')
    # etc
    parser.add_argument('--model_time', type=str, help='automatically set when build model or manually set when load_model is True')
    parser.add_argument('--load_model', type=str2bool, default=False, help='use checkpoint epoch or not')
    parser.add_argument('--checkpoint_epoch', type=int, default=0, help='restart train checkpoint')
    parser.add_argument('--seed', type=int, default=100, help='random seed')
    parser.add_argument('--task', type=str, default='lightweight_sr')
    # training data
    parser.add_argument('--data_name', type=str, default='DIV2K', help='training dataset')
    parser.add_argument('--training_patch_size', type=int, default=64, help='LQ image patch size. model input patch size only for training')
    # model type
    parser.add_argument('--model_name', type=str, default='NGswin', help='NGswin or SwinIR-NG')
    parser.add_argument('--target_mode', type=str, default='light_x2', help='light_x2 light_x3 light_x4')
    parser.add_argument('--scale', type=int, help="upscale factor corresponding to 'target_mode'. it is automatically set.")
    parser.add_argument('--window_size', type=int, default=8, help='window size of (shifted) window attention')
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
    parser.add_argument('--total_epochs', type=int, default=500, help='number of total epochs')
    parser.add_argument('--test_only', type=str2bool, default=False, help='only evaluate model. not train')
    parser.add_argument('--test_epoch', type=int, default=20, help='each epoch to run model evaluation')
    parser.add_argument('--sr_image_save', type=str2bool, default=True, help='save reconstructed image at test')
    # dataset / dataloader spec
    parser.add_argument('--img_norm', type=str2bool, default=True, help="image normalization before input") # -> ignored for SwinIR-NG
    parser.add_argument('--batch_size', type=int, default=64, help="mini-batch size assigned to total gpus")
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers in dataloader")
    parser.add_argument('--pin_memory', type=str2bool, default=True, help="pin-memory in dataloader")
    parser.add_argument('--record_iter', type=int, default=100, help='iteration to record history while training')
    # optimizer optim spec
    parser.add_argument('--lrd', type=str, default='half', help="learning rate decay schedule")
    parser.add_argument('--init_lr', type=float, default=4e-4, help='base initial learning rate')
    parser.add_argument('--min_lr', type=float, default=2e-06, help='minimum learning rate only for cosine-decay')
    parser.add_argument('--warmup_epoch', type=int, default=20, help='warmup epoch')
    parser.add_argument('--accum_iter', type=int, default=1, help='full-batch size divided by mini-batch size')
    parser.add_argument('--lr_cycle', type=int, help='one frequency of cosine learning rate decay. it is automatically set.')
    parser.add_argument('--half_list', type=str2tuple, default=(200, 300, 400, 425, 450, 475), help='epochs at which learning rate is reduced in half, if lrd is half')
    parser.add_argument('--criterion', type=str, default='L1', help='loss function')
    # optimzier regularizer / criterion spec
    # negative effects -> no use
    parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay')
    parser.add_argument('--layer_decay', type=float, default=1.0, help='layer-wise learning rate decay. 1.0 means no lwd')
    parser.add_argument('--max_norm', type=float, help='clip grad max norm')
    
    parser.add_argument('--nparams', type=int, help='automatically set')
    parser.add_argument('--macs', type=float, help='automatically set')
    
    return parser

def main(args):
    torch.manual_seed(args.seed)
    train_loader = build_dataset(args)
    model, optimizer, loss_scaler = build_model_optimizer_scaler(args)
    criterion = build_loss_func(args)
    
    args = record_args_after_build(args)
    for epoch in range(args.total_epochs):
        if (epoch+1) <= args.checkpoint_epoch: continue
        if (epoch+1) == args.warm_start_epoch+1:
            model, optimizer, loss_scaler = rebuild_after_warm_start(model, args)
            record_whole_finetune_time(args)
        if not args.test_only:
            train_one_epoch(model, train_loader, optimizer, loss_scaler, criterion, epoch, (epoch+1)<=args.warm_start_epoch, args)
        if (epoch+1)%args.test_epoch==0 or (epoch+1)==args.total_epochs:
            test_one_epoch(model, epoch, args)

if __name__ == '__main__':
    os.makedirs(f'./args/', exist_ok=True)
    os.makedirs(f'./logs/', exist_ok=True)
    parser = get_args_parser()
    args = parser.parse_args()
    
    args.scale = int(args.target_mode[-1]) if args.target_mode[-1].isdigit() else 1
    args.lr_cycle = args.warm_start_epoch
    
    args = record_args_before_build(args)
    main(args)