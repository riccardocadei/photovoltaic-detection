import torch.utils.data as data
from torchvision.transforms import transforms
import glob
import numpy as np
import os
from PIL import Image
import torch


class DataLoaderSegmentation(data.Dataset):
    def __init__(self, folder_path_img,folder_path_mask):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path_img,'*.png'))
        self.mask_files =glob.glob(os.path.join(folder_path_mask,'*.png'))
        self.transformsimage = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),transforms.ToTensor()])
        self.transformsmask = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),transforms.ToTensor()])

    def __getitem__(self, index):
        image = Image.open(self.img_files[index])
        mask = Image.open(self.mask_files[index])
        t_image,t_mask = self.transformsimage(image),self.transformsmask(mask)
        t_mask = np.asarray(t_mask)
        t_mask = t_mask[0]
        t_mask = t_mask > 0
        t_mask = torch.tensor(t_mask, dtype = torch.float)
        return t_image,t_mask

    def __len__(self):
        return len(self.img_files)