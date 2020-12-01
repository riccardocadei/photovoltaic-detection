import torch.utils.data as data
from torchvision.transforms import transforms
import glob
import numpy as np
import os
from PIL import Image
import torch
import random
# import  data_loader.flip 
# import data_loader.normalization2


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