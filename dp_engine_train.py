import torch

import os
import datetime
from tqdm import tqdm

from utils.train_utils import cosine_learning_rate, half_learning_rate, keep_num_files

# Super-Resolution
def train_one_epoch(model, data_loader, optimizer, loss_scaler, criterion, epoch, args):
    epoch_zfill = len(str(args.total_epochs))
    iter_zfill = len(str(len(data_loader)))
    
    model.train()
    total_loss = 0
    for data_iter, (img_hr, img_lr) in enumerate(tqdm(data_loader)):
                    
        if data_iter%args.accum_iter==0:
            if args.lrd=='cosine':
                cosine_learning_rate(optimizer, epoch + data_iter/len(data_loader), args)
            elif args.lrd=='half':
                half_learning_rate(optimizer, epoch + data_iter/len(data_loader), args)
        
        img_hr = img_hr.to(args.device)
        img_lr = img_lr.to(args.device)
        
        with torch.cuda.amp.autocast():
            img_sr = model(img_lr)
            loss = criterion(img_sr, img_hr)
            
        loss /= args.accum_iter
        loss_scaler(loss, optimizer, clip_grad=args.max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter+1)%args.accum_iter==0)
        
        if (data_iter+1)%args.accum_iter==0:
            optimizer.zero_grad()
            
        total_loss += loss.item()*args.accum_iter
        
        if (data_iter+1)%args.record_iter==0 and (data_iter+1)!=len(data_loader):
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(f'./logs/{args.model_time}_train.txt', 'a') as f:
                f.writelines(f'epoch: [{str(epoch+1).zfill(epoch_zfill)}/{args.total_epochs}], ')
                f.writelines(f'iter: [{str(data_iter+1).zfill(iter_zfill)}/{len(data_loader)}], ')
                f.writelines(f'loss: {total_loss/((data_iter+1)):.8f} {now}\n')
    
    # end of train one epoch
    avg_loss = total_loss/len(data_loader)
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(f'./logs/{args.model_time}_train.txt', 'a') as f:
        f.writelines(f'epoch: [{str(epoch+1).zfill(epoch_zfill)}/{args.total_epochs}], ')
        f.writelines(f'iter: [{str(data_iter+1).zfill(iter_zfill)}/{len(data_loader)}], ')
        f.writelines(f'loss: {avg_loss:.8f} {now}\n')

    # model, optimizer, scaler state_dict SAVE
    sd_save_list = ['models', 'optims', 'scalers']
    for sd_save in sd_save_list: os.makedirs(f'./{sd_save}/{args.model_time}', exist_ok=True)
    if 'module' in model.__dir__(): torch.save(model.module.state_dict(), f'./models/{args.model_time}/model_{str(epoch+1).zfill(epoch_zfill)}.pth')
    else: torch.save(model.state_dict(), f'./models/{args.model_time}/model_{str(epoch+1).zfill(epoch_zfill)}.pth')
    torch.save(optimizer.state_dict(), f'./optims/{args.model_time}/optim_{str(epoch+1).zfill(epoch_zfill)}.pth')
    torch.save(loss_scaler.state_dict(), f'./scalers/{args.model_time}/scaler_{str(epoch+1).zfill(epoch_zfill)}.pth')
    for sd_save in sd_save_list: keep_num_files(f'./{sd_save}/{args.model_time}', 'pth', 300)