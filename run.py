
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
    torch.manual_seed(0)
    np.random.seed(0)
    folder_path_image = 'data/image'
    folder_path_mask  = 'data/mask'
    folder_path_noara  = 'data/noARA'

    #load dataset
    train_dataset = ConcatDataset([DataLoaderSegmentation(folder_path_image,folder_path_mask),DataLoaderNoARA(folder_path_noara)])

    #combine two datasets
    train_loader = DataLoader(train_dataset,batch_size=5, shuffle=True ,num_workers=0)
    # %%

    lr_candidates = np.logspace(-2,-3,3)
    num_epochs = 70
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([6]).cuda())
    input_model = UNet(3,1,False).to(device)

    best_lr, best_model, best_iou = select_hyper_param(train_dataset,loss_function,input_model,num_epochs,lr_candidates)

    torch.save(best_model.state_dict(), 'model/best_model.pt')
 