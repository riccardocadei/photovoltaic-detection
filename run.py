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
    train_loader = DataLoader(ConcatDataset([train_dataset,noara_dataset]),batch_size=5, shuffle=True ,num_workers=0)

    # %% [markdown]
    # # Initiate the model
    # In this report, we will use the Unet model presented in medical image segmentation, and in the previous papers of the Professor.

    # %%
    model = UNet(3,1,False).to(device)
    print(model)

    # %% [markdown]
    # # Loss & Optimizer

    # %%
    loss_function = torch.nn.BCELoss(weight=torch.FloatTensor([6]).cuda())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # %% [markdown]
    # # Training Loop

    # %%
    num_epochs = 200
    model = UNet(3,1,False).to(device)

    trained_model = training_model(train_loader,loss_function,optimizer,model,num_epochs)

    # %%
    torch.save(model.state_dict(), 'model/trained_model.pt')

