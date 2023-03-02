# reference
# Masked Autoencoders Are Scalable Vision Learners https://arxiv.org/abs/2111.06377
# BEiT: BERT Pre-Training of Image Transformers https://arxiv.org/abs/2106.08254

import torch
import math
from torch._six import inf
import os
import numpy as np
import re

def keep_num_files(root, extension, num):
    files = sorted([x for x in os.listdir(root) if x.endswith(extension)], reverse=True)
    while len(files) > num:
        os.remove(f'{root}/{files[-1]}')
        files.pop()
        
# cosine learning rate schdule
def cosine_learning_rate(optimizer, epoch, args):
    """
    Decay the learning rate with half-cycle cosine after warmup
    Following MAE: https://arxiv.org/abs/2111.06377
    """
    if args.lr_cycle is None: args.lr_cycle = args.total_epochs
    epoch = epoch if epoch < args.lr_cycle else epoch-args.lr_cycle
    if epoch < args.warmup_epoch:
        lr = args.init_lr * epoch / args.warmup_epoch
    else:
        lr = args.min_lr + (args.init_lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epoch) / (args.total_epochs - args.warmup_epoch)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

# half learning rate schdule
def half_learning_rate(optimizer, epoch, args, warm_start=False):
    if args.lr_cycle is None: args.lr_cycle = args.total_epochs
    epoch = epoch if epoch < args.lr_cycle else epoch-args.lr_cycle
    if warm_start:
        lr = args.init_lr * epoch / args.warmup_epoch
        return lr
    
    if epoch < args.warmup_epoch:
        lr = args.init_lr * epoch / args.warmup_epoch
    else:
        half_list = np.array(args.half_list)
        ndecay = (half_list<epoch).sum()
        lr = args.init_lr*(0.5**ndecay)
        if epoch%1 == 0.0:
            print(epoch, lr)
            
    for param_group in optimizer.param_groups:            
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

# layer-wise learning rate decay
def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75, model_name='NGswin'):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT https://arxiv.org/abs/2106.08254
    """
    assert model_name in ['NGswin', 'SwinIR-NG'], "'model_name' should be NGswin or SwinIR-NG"
    param_group_names = {}
    param_groups = {}
        
    if model_name=='NGswin':
        swin_blocks = [-1 for _ in range(6)] # encoder(4stages) / decoder(2stages) for extra-stages. default: e 3stages / d 1stage
        for n, p in model.named_parameters():
            if n.startswith('encoder'):
                # n ex: encoder_layer1.blocks.1.mlp.fc2.bias / encoder_layer1.downsample.norm.weight
                layer_idx, level, block_idx = re.search('encoder_layer(\d)\.([^\.]+)\.([^\.]+)',n).groups()
                layer_idx = int(layer_idx)-1
                block_idx = int(block_idx) if block_idx.isdigit() else swin_blocks[layer_idx]
                if block_idx != swin_blocks[layer_idx]:
                    swin_blocks[layer_idx] += 1
            elif n.startswith('decoder'):
                layer_idx, level, block_idx = re.search('decoder_layer(\d)\.([^\.]+)\.([^\.]+)',n).groups()
                layer_idx = int(layer_idx)-1
                block_idx = int(block_idx) if block_idx.isdigit() else swin_blocks[4+layer_idx]
                if block_idx != swin_blocks[4+layer_idx]:
                    swin_blocks[4+layer_idx] += 1
    else:
        swin_blocks = [-1 for _ in range(4)] # 4layers
        for n,p in model.named_parameters():
            if n.startswith('layers'):
                layer_idx, level = re.search('layers\.(\d)\.([^\.]+)\.', n).groups()
                layer_idx = int(layer_idx)
                block_idx = int(re.search('layers\.\d\.[^\.]+\.blocks\.(\d)', n).group(1)) if level == 'residual_group' else swin_blocks[layer_idx]
                if block_idx != swin_blocks[layer_idx]:
                    swin_blocks[layer_idx] += 1

    swin_blocks = [x+1 for x in swin_blocks]
    num_layers = sum(swin_blocks)+2
        
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))
    
    for n, p in model.named_parameters():
        if not p.requires_grad: continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
        
        if model_name=='NGswin':
            layer_id = get_layer_id_for_ngswin(n, num_layers, swin_blocks)
        else:
            layer_id = get_layer_id_for_swinirng(n, num_layers, swin_blocks)
        group_name = f'layer_{layer_id}_{g_decay}'
        
        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
                "param_names": []
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)
        param_groups[group_name]["param_names"].append(n)
    
    return list(param_groups.values())

def get_layer_id_for_ngswin(name, num_layers, swin_blocks):
    """
    Assign a parameter with its layer id
    Referring to BEiT and MAE
    """
    if name.startswith('shallow'):
        return 0
        
    elif name.startswith('bottleneck'):
        return sum(swin_blocks[:4])+1
    elif name.startswith(('encoder', 'decoder')):
        if name.startswith('e'):
            layer_idx, level, block_idx = re.search('encoder_layer(\d)\.([^\.]+)\.([^\.]+)',name).groups()
            layer_idx = int(layer_idx)-1
            block_idx = int(block_idx) if block_idx.isdigit() else 0 if 'cascade' in name else swin_blocks[layer_idx]-1
        else:
            layer_idx, level, block_idx = re.search('decoder_layer(\d)\.([^\.]+)\.([^\.]+)',name).groups()
            layer_idx = 4+int(layer_idx)-1
            block_idx = int(block_idx) if block_idx.isdigit() else 0 if 'cascade' in name else swin_blocks[layer_idx]-1
        idx = 0
        for i in range(layer_idx):
            idx += swin_blocks[i]
        idx += block_idx+1 if 'encoder' in name else block_idx+2
        return idx
    elif name.startswith('norm'):
        return num_layers-1
    else:
        return num_layers
    
def get_layer_id_for_swinirng(name, num_layers, swin_blocks):
    if name.startswith(('conv_first', 'patch_embed')):
        return 0
    
    elif name.startswith('layers'):
        layer_idx, level = re.search('layers\.(\d)\.([^\.]+)\.', name).groups()
        layer_idx = int(layer_idx)
        block_idx = int(re.search('layers\.\d\.[^\.]+\.blocks\.(\d)', name).group(1)) if level == 'residual_group' else swin_blocks[layer_idx]-1
        idx = 0
        for i in range(layer_idx):
            idx += swin_blocks[i]
        idx += block_idx+1
        return idx
    elif name.startswith('norm'):
        return num_layers-1
    else:
        return num_layers
    
    return
    
# loss scaler
class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm