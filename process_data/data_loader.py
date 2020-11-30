import torch.utils.data as data
from torchvision.transforms import transforms
import glob
import numpy as np
import os
from PIL import Image
import torch
import random


class DataLoaderSegmentation(data.Dataset):
    
    def __init__(self, folder_path_img,folder_path_mask):
        """
        Args:
            image_path (str): the path where the image is located
            mask_path (str): the path where the mask is located
            option (str): decide which dataset to import
        """
        self.img_files = glob.glob(os.path.join(folder_path_img,'*.png'))
        self.mask_files =glob.glob(os.path.join(folder_path_mask,'*.png'))

        
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

        # AUGMENTATION
        
        # flip {0: vertical, 1: horizontal, 2: both, 3: none}
        flip_num = random.randint(0, 3) 
        img_as_np = flip(img_as_np, flip_num)
        # Normalize the image (in min max range)
        img_as_np = normalization2(img_as_np, max=1, min=0)
       
        # Convert numpy array to tensor
        img_as_np = np.transpose(img_as_np,(2,0,1))
        img_as_tensor = torch.from_numpy(img_as_np).float()  
        
        
        """
        # GET MASK
        """
        mask = Image.open(self.mask_files[index])
        msk_as_np = np.asarray(mask)
        
        # AUGMENTATION
        
        # flip the mask with respect to image
        msk_as_np = flip(msk_as_np, flip_num)
        
        msk_as_np = np.transpose(msk_as_np,(2,0,1))
        msk_as_np = msk_as_np[0]
        msk_as_np = msk_as_np > 0
        
        # Convert numpy array to tensor
        msk_as_tensor =  torch.tensor(msk_as_np, dtype = torch.float)
        
        return img_as_tensor, msk_as_tensor

    def __len__(self):
        return len(self.img_files)
    
    
    
    
def flip(image, option_value):
    """
    Args:
        image : numpy array of image
        option_value = random integer between 0 to 3
    Return :
        image : numpy array of flipped image
    """
    if option_value == 0:
        # vertical
        image = np.flip(image, option_value)
    elif option_value == 1:
        # horizontal
        image = np.flip(image, option_value)
    elif option_value == 2:
        # horizontally and vertically flip
        image = np.flip(image, 0)
        image = np.flip(image, 1)
    else:
        # no effect
        image = image
        
    return image


def normalization2(image, max, min):
    """Normalization to range of [min, max]
    Args :
        image : numpy array of image
        mean :
    Return :
        image : numpy array of image with values turned into standard scores
    """
    image_new = (image - np.min(image))*(max - min)/(np.max(image)-np.min(image)) + min
    
    return image_new