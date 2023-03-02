import torch
from torchvision.transforms import functional as TF

import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

from utils import acc_utils

@torch.no_grad()
def test_one_epoch(rank, gpu, model, epoch, args):
    model.eval()
    if args.task == 'lightweight_sr':
        default_dataset_list = ['Set5', 'Set14', 'BSDS100', 'urban100', 'manga109']
    min_multiple = (4*args.window_size, 4*args.window_size) if args.model_name=='NGswin' else (args.window_size, args.window_size)
    
    if args.world_size==4:
        dataset_list = [default_dataset_list[rank+1]]
        if rank==0:
            dataset_list.insert(0, default_dataset_list[0])

    elif args.world_size==2:
        if rank==0:
            dataset_list = default_dataset_list[:4]
        else:
            dataset_list = default_dataset_list[4:]

    elif args.world_size==1:
        dataset_list = default_dataset_list
        
    for dd, dataset in enumerate(dataset_list):
        folder_lq = f'../testsets/{dataset}/LR_bicubic/X{args.scale}/'
        folder_hq = f'../testsets/{dataset}/HR/'
        save_dir = f'results/{args.model_name}_{args.task}_x{args.scale}/{args.model_time}'
        border = args.scale if args.task == 'lightweight_sr' else 0
        
        # setup result dictionary
        os.makedirs(save_dir, exist_ok=True)
        test_results = OrderedDict()
        test_results['psnr_y'] = []
        test_results['ssim_y'] = []
        psnr_y, ssim_y = 0, 0
        
        path_list = sorted(glob.glob(os.path.join(folder_hq, '*.npy')))
        imgname_maxlen = max([len(os.path.splitext(os.path.basename(p))[0]) for p in path_list])
        for data_iter, path in enumerate(tqdm(path_list)):
            # read image
            img_hq = torch.from_numpy(np.load(path))/255
            imgname, imgext = os.path.splitext(os.path.basename(path))
            path_lq = os.path.join(folder_lq, f'{imgname}x{args.scale}{imgext}')
            img_lq = torch.from_numpy(np.load(path_lq)).unsqueeze(0)/255
            
            # inference
            # pad input image to be a multiple of window_size X final_patch_size
            _, _, lqh, lqw = img_lq.size()
            padw = min_multiple[1] - (lqw%min_multiple[1]) if lqw%min_multiple[1]!=0 else 0
            padh = min_multiple[0] - (lqh%min_multiple[0]) if lqh%min_multiple[0]!=0 else 0
            img_lq = TF.pad(img_lq, (0,0,padw,padh), padding_mode='symmetric')
            img_rc = model(img_lq.to(gpu))
            img_rc = img_rc[..., :lqh*args.scale, :lqw*args.scale]
            
            # save image
            img_rc = img_rc[0].detach().cpu().clamp(0,1).numpy() if args.img_norm \
                    else img_rc[0].detach().cpu().clamp(0,1).numpy()
            img_rc = np.transpose(img_rc[[2, 1, 0],:,:], (1, 2, 0)) if img_rc.ndim == 3 else img_rc # CHW-RGB to HWC-BGR
            img_rc = (img_rc * 255.0).round().astype(np.uint8)  # float32 to uint8
            if args.sr_image_save:
                cv2.imwrite(f'{save_dir}/{dataset}_{imgname}_x{args.scale}_{args.model_name}.png', img_rc)
            
            # evaluate psnr/ssim
            img_hq = img_hq.permute(1,2,0)[:,:,[2,1,0]].numpy() # CHW-RGB to HWC-BGR
            img_hq = (img_hq * 255.0).round().astype(np.uint8) # float32 to uint8
            img_hq = img_hq[:lqh*args.scale,:lqw*args.scale,:]  # crop HQ
            img_hq = np.squeeze(img_hq)

            psnr_y = acc_utils.calculate_psnr(img_rc, img_hq, crop_border=border, test_y_channel=True)
            ssim_y = acc_utils.calculate_ssim(img_rc, img_hq, crop_border=border, test_y_channel=True)
            test_results['psnr_y'].append(psnr_y)
            test_results['ssim_y'].append(ssim_y)
            with open(f'./logs/{args.model_time}_test_{dataset}.txt', 'a') as f:
                if dd==0 and data_iter==0: f.writelines(f'[[{epoch+1}]]\n')
                for _ in range(imgname_maxlen-len(imgname)): imgname+=' '
                f.writelines(f'{dataset:10s} {data_iter+1:3d} {imgname} - ')
                f.writelines(f'PSNR_Y: {psnr_y:.2f}, SSIM_Y: {ssim_y:.4f}\n')
                if data_iter+1 == len(path_list): f.writelines('\n')
                            
        # summarize psnr/ssim
        with open(f'./logs/{args.model_time}_test.txt', 'a') as f:
            if rank==0 and dd==0: 
                f.writelines(f'[[{epoch+1}]]\n')
            avg_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            avg_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            f.writelines(f'{dataset} {args.target_mode} - PSNR_Y/SSIM_Y: {avg_psnr_y:.2f}/{avg_ssim_y:.4f}\n')
            if rank==args.world_size-1 and dd+1==len(dataset_list):
                f.writelines('\n')