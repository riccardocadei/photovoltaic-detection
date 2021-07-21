from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, dataset
from loss.loss import *
from model.unet import *
from train.train import *

import numpy as np
import time
from torch.autograd import Variable
import torch
import os

def seed_torch(seed=0):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    

def test_model(test_loader,model):
    """ Evaluate the model on a training set, with reported error the mean IoU

    Args:
        test_loader (DataLoader): Training set on which the model will be evaluated on
        model (Model): Model to be tested

    Returns:
        FLoat: Mean IoU on tested set
        Flaot: Mean Accuracy on tested set
    """
    
    iou_test = []
    acc_test = []
    recall_test = []
    precision_test = []
    for i, (images,labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())

        for i in range(images.shape[0]):
            prediction = model.predict(torch.unsqueeze(images[i],0))
            iou_i = iou(np.around((prediction).detach().cpu().numpy()),labels[i].detach().cpu().numpy())
            iou_test.append(iou_i)
            acc_i = accuracy(np.around((prediction).detach().cpu().numpy()),labels[i].detach().cpu().numpy())
            acc_test.append(acc_i)
            recall_i = recall(np.around((prediction).detach().cpu().numpy()),labels[i].detach().cpu().numpy())
            recall_test.append(recall_i)
            precision_i = precision(np.around((prediction).detach().cpu().numpy()),labels[i].detach().cpu().numpy())
            precision_test.append(precision_i)
    return [np.mean(iou_test), np.mean(acc_test), np.mean(recall_test), np.mean(precision_test)]





def select_hyper_param(train_dataset,n_splits,loss_function,num_epochs,lr_candidates):
    """Performs a grid search on a range of learning rates to find the best lr in term of the mean iou on validation. To have an estimate
    of the mean iou we perform cross validation for each learning rate.

    Args:
        train_dataset (Dataset): Data that will be splited in k folds for training and validation
        n_splits (Int): Number of folds (i.e k)
        loss_function (LossFun): Loss Function
        num_epochs (Int): Number of iterations the models will be trained on at each fold  
        lr (float): Learning rate for the optimizer
        lr_candidates (list/np.array): Range on learning rates for grid search

    Returns:
        float: Best Learning Rate obtained
        float: Best IoU obtained
    """
    
    comparison = []

    for lr in lr_candidates:
        print('---------------------------------------------------------------------\n')
        print('Learning Rate = {}\n'.format(lr))
        iou, acc = cross_validation(train_dataset,n_splits, loss_function, num_epochs, lr)
        comparison.append([lr, iou, acc])
    comparison = np.array(comparison).reshape(len(lr_candidates),4)
    ind_best =  np.argmax(comparison[:,1]) 
    best_lr = comparison[ind_best,0]
    best_iou = np.max(comparison[:,1])
        
    return best_lr, best_iou


def cross_validation(train_dataset, n_splits, loss_function,num_epochs,lr):
    """Performs k-fold cross validation on the train set. This function is used to predict how good a model trained on the training set
    will be on the test set.

    Args:
        train_dataset (Dataset): Data that will be splited in k folds for training and validation
        n_splits (Int): Number of folds (i.e k)
        loss_function (LossFun): Loss Function
        num_epochs (Int): Number of iterations the models will be trained on at each fold  
        lr (float): Learning rate for the optimizer

    Returns:
        float: Mean IoU on Validations
        float: Mean Accuracy on Validations
    """
    
    iou_val = []
    acc_val = []
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #define kfold
    kfold =KFold(n_splits=2,shuffle=True)
    for fold, (train_index, test_index) in enumerate(kfold.split(train_dataset)): 
        model = UNet(3,1,False).to(device)
        # split into k Folders
        train_fold = dataset.Subset(train_dataset,train_index)
        test_fold = dataset.Subset(train_dataset,test_index) 
        train_fold_loader = DataLoader(train_fold,batch_size=2, shuffle=True,num_workers=2)
        test_fold_loader = DataLoader(test_fold,batch_size=2, shuffle=True,num_workers=2)
        
        #train the model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        training_model(train_fold_loader,loss_function,optimizer,model,num_epochs)
        # make prediction and compute the evaluation metrics
        iou, acc = test_model(test_fold_loader,model)
        print('Iter {}: IoU = {:.4} /  Accuracy = {:.4}'.format(fold, iou, acc))
        iou_val.append(iou)
        acc_val.append(acc)
        
    print("\nAverage validation IoU: %f" % np.mean(iou_val))
    print("Variance validation IoU: %f" % np.var(iou_val))
    print("\nAverage validation accuracy: %f" % np.mean(acc_val))
    print("Variance validation accuracy: %f" % np.var(acc_val))
        
    return np.mean(iou_val), np.mean(acc_val)


def adptative_learning(train_dataset, val_loader,loss_function,input_model,num_epochs,lr_candidates):
    """Performs a training on a model over a training data set by doing the following: we first fix the learing rate, then we split the training set
    into two folds, the model is trained on the first fold then on the second fold. After this has been done, we move on the next learning rate.
    Contrarly to select_hyper_param, we do not reset the model and we continuously evaluate the iou on the validation set.

    Args:
        train_dataset (Dataset): Data on which that will be splitted and the model solely trained on 
        val_loader (Dataloader): Data used only for tracking the performance of the model
        loss_function (LossFun): Loss Function
        input_model (Model): Model to be trained 
        num_epochs (Int): Number of iterations the models will be trained on at each fold  
        lr_candidates (List): Range of learning rates for the optimizer

    Returns:
        Model: Final Trained Model
        Float : Best IoU
        List : History of loss on validaiton
    """
    
    comparison = []

    for lr in lr_candidates:
        print('---------------------------------------------------------------------\n')
        print('Learning Rate = {}\n'.format(lr))
        iou, acc = adptative_helper(train_dataset,val_loader, loss_function, input_model, num_epochs, lr)
        comparison.append([lr, iou, acc])
    comparison = np.array(comparison).reshape(len(lr_candidates),4)
    ind_best =  np.argmax(comparison[:,1]) 
    best_lr = comparison[ind_best,0]
    best_iou = np.max(comparison[:,1])
        
    return best_iou, comparison[:,1]

def adptative_helper(train_dataset, val_loader, loss_function,input_model,num_epochs,lr):
    
    iou_val = []
    acc_val = []
    #define kfold
    kfold =KFold(n_splits=2,shuffle=True)
    for fold, (train_index, test_index) in enumerate(kfold.split(train_dataset)): 
        # split into k Folders
        train_fold = dataset.Subset(train_dataset,train_index)
        train_fold_loader = DataLoader(train_fold,batch_size=2, shuffle=True,num_workers=2)
        
        #train the model
        optimizer = torch.optim.Adam(input_model.parameters(), lr=lr)
        training_model(train_fold_loader,loss_function,optimizer,input_model,num_epochs)
        
        # make prediction and compute the evaluation metrics
        iou, acc = test_model(val_loader,model)
        print('Iter {}: IoU = {:.4} /  Accuracy = {:.4}'.format(fold, iou, acc))
        iou_val.append(iou)
        acc_val.append(acc)
        
    print("\nAverage validation IoU: %f" % np.mean(iou_val))
    print("Variance validation IoU: %f" % np.var(iou_val))
    print("\nAverage validation accuracy: %f" % np.mean(acc_val))
    print("Variance validation accuracy: %f" % np.var(acc_val))
        
    return np.mean(iou_val), np.mean(acc_val)