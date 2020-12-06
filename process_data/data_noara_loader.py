import torch.utils.data as data
from torchvision.transforms import transforms
import glob
import numpy as np
import os
from PIL import Image
import torch
import random
from process_data.data_loader import flip, normalization2
import matplotlib.pyplot  as plt

class DataLoaderNoARA(data.Dataset):
    """
    load noPV data, only pictures and  mask are all 0(black)
    """
    def __init__(self, folder_path_img):
        """
        Args:
            image_path (str): the path where the image is located
            option (str): decide which dataset to import
        """
        self.img_files = glob.glob(os.path.join(folder_path_img,'*.png'))

        
    def __getitem__(self, index):
        """Get specific data corresponding to the index applying randomly dat augmentation
        Args:
            index (int): index of the data
        Returns:
            Tensor: specific data on index which is converted to Tensor
        """
        
        """
        # GET IMAGE
        """
        image = Image.open(self.img_files[index])
        img_as_np = np.asarray(image)
        np.random.seed(0)

        # AUGMENTATION
        
        # flip {0: vertical, 1: horizontal, 2: both, 3: none}
        flip_num = random.randint(0, 3)         
        img_as_np = flip(img_as_np, flip_num)
        # rotate of rot_num*90 degrees in counterclockwise
        # since we are altready flipping, rotating of 180 or 270 is redundant
        rot_num = random.randint(0, 1)
        img_as_np = np.rot90(img_as_np, rot_num)
        # add noise {0: Gaussian_noise, 1: uniform_noise, 2: no_noise}
        #noise_num = random.randint(0, 2)
        #noise_param = 20
        #img_as_np = add_noise(img_as_np, noise_num, noise_param)
        # Brightness and Saturation
        sat = random.randint(0,75)
        bright = random.randint(0,40)
        img_as_np = change_hsv(img_as_np, sat, bright)
       
        # Convert numpy array to tensor
        img_as_np = np.transpose(img_as_np,(2,0,1))
        img_as_tensor = torch.from_numpy(img_as_np).float()  
        


        """
        # GET noPv MASK All balck
        """

        #Cut channel for binary pictures
        mask_shape = (img_as_np.shape[1],img_as_np.shape[2]) 
        mask = np.zeros(mask_shape)
        
        # Convert numpy array to tensor
        msk_as_tensor =  torch.tensor(mask, dtype = torch.float)
        
        return img_as_tensor, msk_as_tensor

    def __len__(self):
        return len(self.img_files)
