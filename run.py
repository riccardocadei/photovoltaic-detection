import numpy as np
import matplotlib.pyplot  as plt
import torch
from torch.autograd import Variable
from torchvision.transforms.functional import normalize
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, ConcatDataset
from train.train import *
from tempfile import TemporaryFile
from process_data.normalize import * 

from model.unet import *
from loss.loss import *
from process_data.data_loader import *
from hyperparameters.select_param import *
from process_data.import_test import *
from plots.plots import *

if __name__ ==  '__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_torch() # For reproducibility we set the seed with a seed_torch() method that set the seed in numpy and pytorch

    folder_path_train_image = 'data/all/train/images'
    folder_path_train_masks = 'data/all/train/labels'
    folder_path_test_image = 'data/all/test/images'
    folder_path_test_masks = 'data/all/test/labels'
    folder_path_val_image = 'data/all/val/images'
    folder_path_val_masks = 'data/all/val/labels'

    # Load dataset
    train_set = DataLoaderSegmentation(folder_path_train_image,folder_path_train_masks) # 80%
    test_set = DataLoaderSegmentation(folder_path_test_image,folder_path_test_masks,augment=False)# 10%, no augmentation
    val_set = DataLoaderSegmentation(folder_path_val_image,folder_path_val_masks,augment=False) # 10%, no augmentation

    # Init data loader
    train_loader = DataLoader(train_set,batch_size=5, shuffle=True ,num_workers=0)
    val_loader = DataLoader(val_set,batch_size=5, shuffle=True ,num_workers=0)
    test_loader = DataLoader(test_set,batch_size=5, shuffle=True ,num_workers=0)

    model = UNet(3,1,False).to(device)
    print(len(train_set),len(test_set),len(val_set))



    # Init training parameters
    num_epochs = (600)
    loss_function = torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([4]).cuda())
    model = UNet(3,1,False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # We opted for the linear scheduler. For example, every 60 epochs the learning rate is multiplied by 0.8.
    al_param=60
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, al_param, gamma=0.8, last_epoch=-1, verbose=False)

    # Train model
    history_train_loss, history_val_loss, history_train_iou, history_val_iou = training_model(train_loader,loss_function,optimizer,model,num_epochs,scheduler,val_loader)

    torch.save(model.state_dict(), 'model/trained_model.pt')
 