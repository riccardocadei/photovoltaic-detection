
import numpy as np
import matplotlib.pyplot  as plt
import torch
import os
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

    folder_path_image = 'data/image_residencial'
    folder_path_mask  = 'data/mask_residencial'
    folder_path_noara  = 'data/noARA'

    #load dataset
    dataset = ConcatDataset([DataLoaderSegmentation(folder_path_image,folder_path_mask),DataLoaderSegmentation(folder_path_noara)])

    #split into train, val, test
    dataset_size = len(dataset)
    train_size = int(0.8*len(dataset))
    val_size = int(0.1*len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])


    train_loader = DataLoader(train_set,batch_size=2, shuffle=True ,num_workers=0)
    val_loader = DataLoader(val_set,batch_size=2, shuffle=True ,num_workers=0)
    test_loader = DataLoader(test_set,batch_size=2, shuffle=True ,num_workers=0)


    num_epochs = 400
    loss_function = torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([6]).cuda())
    model = UNet(3,1,False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.63, last_epoch=-1, verbose=False)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0.01, last_epoch=-1, verbose=True)

    history_train_loss, history_val_loss, history_train_iou, history_val_iou = training_model(train_loader,loss_function,optimizer,model,num_epochs,scheduler)

    torch.save(model.state_dict(), 'model/trained_model.pt')
 