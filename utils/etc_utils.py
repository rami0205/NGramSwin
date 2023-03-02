import torch
import torch.nn as nn

class Denormalize(nn.Module):
    def __init__(self, mean, std):
        super(Denormalize, self).__init__()
        self.mean = mean
        self.std = std
        
    def forward(self, img):
        return denormalize(img, self.mean, self.std)
    
def denormalize(img, mean, std):
    assert isinstance(mean, tuple), 'mean and std should be tuple'
    assert isinstance(std, tuple), 'mean and std should be tuple'
    
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    if img.ndim==4:
        mean, std = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).type_as(img), std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).type_as(img)
    elif img.ndim==3:
        mean, std = mean.unsqueeze(-1).unsqueeze(-1).type_as(img), std.unsqueeze(-1).unsqueeze(-1).type_as(img)
    return img*std+mean