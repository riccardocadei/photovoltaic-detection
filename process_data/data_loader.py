import torch.utils.data as data
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
import glob
import numpy as np
import os
from PIL import Image
import torch
import random
import cv2


class DataLoaderSegmentation(data.Dataset):
    """Class for all three data loaders (train, test, val)
    """
    
    def __init__(self, folder_path_img,folder_path_mask=None,augment=True):
        """
        Args:
            image_path (str): the path where the image is located
            mask_path (str): the path where the mask is located
            option (str): decide which dataset to import
        """
        self.img_files = glob.glob(os.path.join(folder_path_img,'*.png'))
        if folder_path_mask == None:
            self.mask_files = 0
        else:
            self.mask_files =glob.glob(os.path.join(folder_path_mask,'*.png'))
        self.augment = augment

    def augmentor(self,image,mask):
        if np.random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
    
        if np.random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        if np.random.random() > 0.5:
            image = TF.rotate(image,90)
            mask = TF.rotate(mask,90)
        
        if random.random() > 0:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(248, 248))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

        if random.random() > 0:
            sigma = np.random.uniform(0,0.05)
            image = TF.gaussian_blur(image, 3, sigma=sigma)

            

        return image, mask

    def transform(self, image, mask):
        # AUGMENTATION


        if random.random() > 0:
            bright = np.random.normal(1,0.1)
            image = TF.adjust_brightness(image,bright)
        
    
        image = TF.to_tensor(image)
        #image = TF.normalize(image,mean=[0.3268, 0.5080, 0.3735],std=[0.2853, 0.2395, 0.2063]) # For Residential
        image = TF.normalize(image,mean=[0.4066, 0.4768, 0.4383],std=[0.2121, 0.1899, 0.1618]) # For All
        #image = TF.normalize(image,mean=[0.3877, 0.5270, 0.4675],std=[0.2972, 0.2621, 0.2189]) # For Industrial
        
        mask = TF.to_tensor(mask)
        mask = mask[0]
        mask = mask > 0
        mask = mask.float()
                
        return image, mask

    def __getitem__(self, index,show_og=False):
        """Get specific data corresponding to the index applying randomly dat augmentation
        Args:
            index (int): index of the data
        Returns:
            Tensor: specific data on index which is converted to Tensor
        """
        
        """
        # GET IMAGE
        """
        image = Image.open(self.img_files[index]).convert('RGB')
        if self.mask_files != 0:
            mask = Image.open(self.mask_files[index])
        else:
            mask_shape = image.size
            mask = Image.new('RGB', mask_shape)        
        
        if self.augment:
            image, mask = self.augmentor(image,mask)

        z = image
        x, y = self.transform(image, mask)

        if show_og:
            return x, y, z
        else:
            return x,y

    

    def __len__(self):
        return len(self.img_files)



def change_hsv(image, sat, bright):
    """
    Args:
        image : numpy array of image
        sat: saturation
        bright : brightness
    Return :
        image : numpy array of image with saturation and brightness added
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            s[i,j]=min(s[i,j]+sat,255)
            v[i,j]=min(v[i,j]+bright,255)
    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return image
    
    
    
    
    
    
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

def add_noise(image, option_value, param):
    if option_value==0:
        # Gaussian_noise
        gaus_sd, gaus_mean = random.randint(0, param), 0
        image = add_gaussian_noise(image, gaus_mean, gaus_sd)
    elif option_value==1:
        # uniform_noise
        l_bound, u_bound = random.randint(-param, 0), random.randint(0, param)
        image = add_uniform_noise(image, l_bound, u_bound)
    else:
        # no noise
        image = image
    return image       

def add_uniform_noise(image, low=-10, high=10):
    """
    Args:
        image : numpy array of image
        low : lower boundary of output interval
        high : upper boundary of output interval
    Return :
        image : numpy array of image with uniform noise added
    """
    uni_noise = np.random.uniform(low, high, image.shape)
    image = image.astype("int16")
    noise_img = image + uni_noise
    image = ceil_floor_image(image) 
    return noise_img

def add_gaussian_noise(image, mean=0, std=1):
    """
    Args:
        image : numpy array of image
        mean : pixel mean of image
        standard deviation : pixel standard deviation of image
    Return :
        image : numpy array of image with gaussian noise added
    """
    gaus_noise = np.random.normal(mean, std, image.shape)
    image = image.astype("int16")
    noise_img = image + gaus_noise
    image = ceil_floor_image(image)
    return noise_img

def ceil_floor_image(image):
    """
    Args:
        image : numpy array of image in datatype int16
    Return :
        image : numpy array of image in datatype uint8 with ceilling(maximum 255) and flooring(minimum 0)
    """
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype("uint8")
    return image
