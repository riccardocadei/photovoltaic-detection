
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

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    folder_path_image = 'data/image'
    folder_path_mask  = 'data/mask'
    folder_path_noara  = 'data/noARA'

    #load dataset
    train_dataset = DataLoaderSegmentation(folder_path_image,folder_path_mask)
    noara_dataset = DataLoaderNoARA(folder_path_noara)

    #combine two datasets
    train_loader = DataLoader(ConcatDataset([train_dataset,noara_dataset]),batch_size=5, shuffle=True ,num_workers=0)

    model = UNet(3,1,False).to(device)
    print(model)
    # %%
    num_epochs = 500
    loss_function = torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([5]).cuda())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1, verbose=True)

    trained_model = training_model(train_loader,loss_function,optimizer,model,num_epochs,scheduler)

    torch.save(trained_model.state_dict(), 'model/trained_model.pt')
