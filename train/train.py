from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, dataset
from loss.loss import *
from model.unet import *
from hyperparameters.select_param import *

import numpy as np
import time
from torch.autograd import Variable
import torch
import os

def training_model(train_loader,loss_function,optimizer,model,num_epochs,scheduler=None,val_loader = None):
    """Simple training loop for a model, on a training set, with respect to a loss function and optimizer. The function can take a scheduler
    for the learning rate. 

    Args:
        train_loader (DataLoader): Data on which the model will be trained on
        loss_function (LossFunction): Loss function
        optimizer (Optimizer): Optimizer
        model (Model): Model that will be trained
        num_epochs (Int): Number of iterations for training
        scheduler (Scheduler, optional): Schedule the learning rate, see pytorch doc. Defaults to None.

    Returns:
        list: History of train loss
        list: History of val loss
        list: History of train IoU
        list: History of val IoU
    """
    if scheduler == None: # Constant scheduler that does not affect the lr
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=1, last_epoch=-1, verbose=False) 
        
    history_train_loss = []
    history_val_loss = []
    history_train_iou = []
    history_val_iou = []
    period = 25
    
    for epoch in range(num_epochs+1):

        
        model.train()
        running_train_loss = 0.0
        running_train_iou = []
        t0 = time.time()
        for i, (images,labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images=Variable(images.cuda())
                labels=Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(torch.squeeze(outputs), torch.squeeze(labels))
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            if epoch % period == 0:
                for j in range(images.shape[0]):
                    iou_j = iou(np.around(sigmoid(outputs[j].detach().cpu().numpy())),labels[j].detach().cpu().numpy())
                    running_train_iou.append(iou_j)
        scheduler.step()


        running_val_loss = 0.0
        running_val_iou = []
        model.eval()
        if (epoch % period)==0:
            if (val_loader != None):
                for i, (images,labels) in enumerate(val_loader):
                    if torch.cuda.is_available():
                                    images=Variable(images.cuda())
                                    labels=Variable(labels.cuda())
                    outputs = model(images)
                    loss = loss_function(torch.squeeze(outputs), torch.squeeze(labels))

                    running_val_loss += loss.item()
                    for j in range(images.shape[0]):
                        iou_j = iou(np.around(sigmoid(outputs[j].detach().cpu().numpy())),labels[j].detach().cpu().numpy())
                        running_val_iou.append(iou_j)
                history_val_loss.append(running_val_loss/(len(val_loader)))
                history_val_iou.append(np.mean(running_val_iou))
                print('Epoch n.',epoch, 'Val Loss',np.around(history_val_loss[-1],2), 'Val Iou',np.mean(running_val_iou))
            history_train_loss.append(running_train_loss/(len(train_loader)))
            history_train_iou.append(np.mean(running_train_iou))
            print('Epoch n.',epoch, 'Train Loss',np.around(history_train_loss[-1],2),'Train Iou',np.mean(running_train_iou),'Time Remaining',np.around((num_epochs-epoch)*(time.time()-t0)/60,1),'min')
            print('----------------------------------------------')
    return history_train_loss, history_val_loss, history_train_iou, history_val_iou
