# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

# %% [markdown]
# # Detecting rooftop available surface for installing PV modules in aerial images using Machine Learning

# %%
import numpy as np
import matplotlib.pyplot  as plt
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, ConcatDataset

from process_data.data_noara_loader import *
from model.unet import *
from loss.loss import *
from process_data.data_loader import *
from process_data.data_noara_loader import *
from hyperparameters.select_param import *
from process_data.import_test import *


if __name__ ==  '__main__':
    # %%
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # %% [markdown]
    # # Loading the Data Set
    # First we load the data set that we will use for training. Each sample is an image with its mask (label). An image is represented as a 3x250x250 array with each of the 3 color chanel being 250x250 pixels. The asssociated mask is a 250x250 array, 

    # %%
    folder_path_image = 'data/image'
    folder_path_mask  = 'data/mask'
    folder_path_noara  = 'data/noARA'

    #load dataset
    train_dataset = DataLoaderSegmentation(folder_path_image,folder_path_mask)
    noara_dataset = DataLoaderNoARA(folder_path_noara)
    #combine two datasets
    print(len(train_dataset),print(noara_dataset))

    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([6]).cuda())
    model = UNet(3,1,False).to(device)

    lr_candidates = np.logspace(-2,-3,3)
    num_epochs = 70
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([6]).cuda())

    input_model = UNet(3,1,False).to(device)

    best_lr, best_model, best_iou = select_hyper_param(train_dataset,loss_function,input_model,num_epochs,lr_candidates)
    torch.save(model.state_dict(), 'model/best_model.pt')

