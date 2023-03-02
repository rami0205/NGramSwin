import os
import re
import glob
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

class DatasetRandomCrop(Dataset):
    def __init__(self, crop_size, data_name=None):
        
        super(DatasetRandomCrop, self).__init__()
        
        self.crop_size = crop_size
    
    def __getitem__(self, idx):
        idx%=len(self.file_list_lr)
        file_hr = self.file_list_hr[idx]
        file_lr = self.file_list_lr[idx]
        
        img_hr = torch.from_numpy(np.load(file_hr))/255
        img_lr = torch.from_numpy(np.load(file_lr))/255
        
        _, lr_h, lr_w = img_lr.size()
        
        # random crop patch
        crop_h = random.choice(range(0,lr_h-self.crop_size))
        crop_w = random.choice(range(0,lr_w-self.crop_size))
        crop_lr = TF.crop(img_lr, crop_h, crop_w, self.crop_size, self.crop_size)
        crop_hr = TF.crop(img_hr, crop_h*self.scale, crop_w*self.scale, self.crop_size*self.scale, self.crop_size*self.scale)
        
        # random horizontal flip, random rotation, image normalization
        random_flip, random_rotate = random.choice(range(2)), random.choice(range(4))
        crop_hr, crop_lr = (TF.hflip(crop_hr), TF.hflip(crop_lr)) if random_flip else (crop_hr, crop_lr)
        crop_hr, crop_lr = TF.rotate(crop_hr, angle=90*random_rotate), TF.rotate(crop_lr, angle=90*random_rotate)
        return crop_hr, crop_lr
    
class DIV2KDatasetRandomCrop(DatasetRandomCrop):
    def __init__(self, root_hr, root_lr, crop_size):
        super(DIV2KDatasetRandomCrop, self).__init__(crop_size, 'DIV2K')
        self.scale = int(re.search('.+(\d)', root_lr).group(1))
        
        self.file_list_hr = sorted(glob.glob(os.path.join(root_hr, '*.npy')))
        self.file_list_lr = sorted(glob.glob(os.path.join(root_lr, '*.npy')))
        
    def __len__(self):
        return len(self.file_list_lr)*80