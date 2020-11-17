import torch.utils.data as data
from torchvision.transforms import transforms
import glob
import os
from PIL import Image

class DataLoaderSegmentation(data.Dataset):
    def __init__(self, folder_path_img,folder_path_mask):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path_img,'*.png'))
        self.mask_files =glob.glob(os.path.join(folder_path_mask,'*.png'))
        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        image = Image.open(self.img_files[index])
        mask = Image.open(self.mask_files[index])
        t_image,t_mask = self.transforms(image),mask
        return t_image,t_mask

    def __len__(self):
        return len(self.img_files)