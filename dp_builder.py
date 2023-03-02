from dataset import DIV2KDatasetRandomCrop
from torch.utils.data import RandomSampler, DataLoader
from my_model.ngswin_model.ngswin import NGswin
from my_model.swinirng import SwinIRNG
from utils.train_utils import param_groups_lrd, NativeScalerWithGradNormCount
import torch
import torch.nn as nn
from collections import OrderedDict

def build_dataset(args):
    
    if args.target_mode[-1].isdigit():
        if args.data_name == 'DIV2K':
            train_data = DIV2KDatasetRandomCrop('../DIV2K/DIV2K_train_HR',
                                                f'../DIV2K/DIV2K_train_LR_bicubic/X{args.scale}/',
                                                crop_size=args.training_patch_size)
        train_sampler = RandomSampler(train_data)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler,
                                  num_workers=args.num_workers, pin_memory=args.pin_memory)
        
    return train_loader

def print_complextity(model, args):
    nparams = 0
    for n,p in model.named_parameters(): nparams+=p.numel()
    h, w = 1280//args.scale, 720//args.scale
    unit = 4*model.window_size if args.model_name=='NGswin' else model.window_size
    padh = unit-(h%unit) if h%unit!=0 else 0
    padw = unit-(w%unit) if w%unit!=0 else 0
    macs = model.flops((h+padh,w+padw))/1e9
    print(f'number of model parameters: {nparams}')
    print(f'MACs: {macs:.2f}G')
    args.nparams = nparams
    args.macs = macs

def build_model_optimizer_scaler(args):
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
    
    # for warm-start and fine-tuning
    if 'pretrain_path' in args.__dict__:
        if args.__dict__['pretrain_path'] is not None and not args.load_model:
            sd = torch.load(args.pretrain_path, map_location='cpu')
            new_sd = OrderedDict()
            for n,p in sd.items():
                if args.model_name=='NGswin':
                    new_sd[n] = p if 'to_target' not in n else model.state_dict()[n]
                else:
                    new_sd[n] = p if 'upsample' not in n else model.state_dict()[n]
            for n,p in model.state_dict().items():
                new_sd[n] = p if n not in new_sd else new_sd[n]
            print(model.load_state_dict(new_sd, strict=False))
        if args.warm_start and args.checkpoint_epoch <= args.warm_start_epoch:
            for n,p in model.named_parameters():
                if args.model_name=='NGswin':
                    p.requires_grad = False if 'to_target' not in n else True
                else:
                    p.requires_grad = False if 'upsample' not in n else True
            
    if args.load_model:
        sd = torch.load(f'./models/{args.model_time}/model_{str(args.checkpoint_epoch).zfill(3)}.pth', map_location='cpu')
        print(model.load_state_dict(sd))
    
    print_complextity(model, args)
    
    model = model.to(args.device)
    model_no_dp = model
    param_groups = param_groups_lrd(model_no_dp, weight_decay=args.weight_decay,
                                    no_weight_decay_list=model_no_dp.no_weight_decay(),
                                    layer_decay=args.layer_decay, model_name=args.model_name)
    model = nn.DataParallel(model, device_ids=[x+int(args.device[-1]) for x in range(args.num_device)])
    
    optimizer = torch.optim.Adam(param_groups, lr=args.init_lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScalerWithGradNormCount()
    
    if args.load_model:
        osd = torch.load(f'./optims/{args.model_time}/optim_{str(args.checkpoint_epoch).zfill(3)}.pth', map_location='cpu')
        ssd = torch.load(f'./scalers/{args.model_time}/scaler_{str(args.checkpoint_epoch).zfill(3)}.pth', map_location='cpu')
        optimizer.load_state_dict(osd)
        loss_scaler.load_state_dict(ssd)
    
    return model, optimizer, loss_scaler

def rebuild_after_warm_start(model, args):
    model = model.module.to(args.device)
    for n,p in model.named_parameters():
        p.requires_grad = True
    model_no_dp = model
    param_groups = param_groups_lrd(model_no_dp, weight_decay=args.weight_decay,
                                    no_weight_decay_list=model_no_dp.no_weight_decay(),
                                    layer_decay=args.layer_decay, model_name=args.model_name)
    model = nn.DataParallel(model, device_ids=[x+int(args.device[-1]) for x in range(args.num_device)])
        
    optimizer = torch.optim.AdamW(param_groups, lr=args.init_lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScalerWithGradNormCount()
    
    return model, optimizer, loss_scaler
    

def build_loss_func(args):
    assert args.criterion in ['L1', 'MSE'], "'criterion' should be L1, MSE"
    criterion_dict = {'L1': nn.L1Loss(), 'MSE': nn.MSELoss()}
    criterion = criterion_dict[args.criterion]
    
    return criterion
