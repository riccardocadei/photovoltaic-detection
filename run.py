
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
    seed_torch()

    def _init_fn():
        np.random.seed(0)

    folder_path_image = 'data/image'
    folder_path_mask  = 'data/mask'
    folder_path_noara  = 'data/noARA'

    #load dataset
    dataset = ConcatDataset([DataLoaderSegmentation(folder_path_image,folder_path_mask),DataLoaderNoARA(folder_path_noara)])

    #split into train, val, test
    dataset_size = len(dataset)
    train_size = int(0.8*len(dataset))
    val_size = int(0.1*len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])


    train_loader = DataLoader(train_set,batch_size=2, shuffle=True ,num_workers=0)
    val_loader = DataLoader(val_set,batch_size=2, shuffle=True ,num_workers=0)
    test_loader = DataLoader(test_set,batch_size=2, shuffle=True ,num_workers=0)


    lr_candidates = np.logspace(-1,-2,num=5)
    num_epochs = 100
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([6]).cuda())

    input_model = UNet(3,1,False).to(device)

    best_model, best_iou, history_iou = adptative_learning(train_set,val_loader,loss_function,input_model,num_epochs,lr_candidates)

    torch.save(best_model.state_dict(), 'model/best_model.pt')
 