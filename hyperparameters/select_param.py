from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, dataset
from loss.loss import *
from model.unet import *

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
    

def training_model(train_loader,loss_function,optimizer,model,num_epochs,scheduler=None):

    if scheduler == None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=1, last_epoch=-1, verbose=False)
    
    for epoch in range(num_epochs):
        running_train_loss = 0.0
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
        scheduler.step()
        if (epoch % 1) == 0:
            print('Epoch n.',epoch, 'Train Loss',np.around(running_train_loss/len(train_loader),4),'Time Remaining',np.around((num_epochs-epoch)*(time.time()-t0)/60,4))
    return model


def test_model(test_loader,model):
    
    iou_test = []
    acc_test = []
    for i, (images,labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
        prediction = model(images)
        iou_i = iou(np.around(sigmoid(prediction.detach().cpu().numpy())),labels.detach().cpu().numpy())
        iou_test.append(iou_i)
        acc_i = accuracy(np.around(sigmoid(prediction.detach().cpu().numpy())),labels.detach().cpu().numpy())
        acc_test.append(acc_i)
    return np.mean(iou_test), np.mean(acc_test)





def select_hyper_param(train_dataset,n_splits,loss_function,num_epochs,lr_candidates):
    
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
        
    return best_iou, comparison[:,1]


def cross_validation(train_dataset, n_splits, loss_function,num_epochs,lr):
    
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
        model = training_model(train_fold_loader,loss_function,optimizer,model,num_epochs)
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
    
    comparison = []

    for lr in lr_candidates:
        print('---------------------------------------------------------------------\n')
        print('Learning Rate = {}\n'.format(lr))
        iou, acc, model = adptative_helper(train_dataset,val_loader, loss_function, input_model, num_epochs, lr)
        comparison.append([lr, iou, acc, model])
    comparison = np.array(comparison).reshape(len(lr_candidates),4)
    ind_best =  np.argmax(comparison[:,1]) 
    best_lr = comparison[ind_best,0]
    best_model = comparison[ind_best,3]
    best_iou = np.max(comparison[:,1])
        
    return best_model, best_iou, comparison[:,1]

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
        model = training_model(train_fold_loader,loss_function,optimizer,input_model,num_epochs)
        
        # make prediction and compute the evaluation metrics
        iou, acc = test_model(val_loader,model)
        print('Iter {}: IoU = {:.4} /  Accuracy = {:.4}'.format(fold, iou, acc))
        iou_val.append(iou)
        acc_val.append(acc)
        
    print("\nAverage validation IoU: %f" % np.mean(iou_val))
    print("Variance validation IoU: %f" % np.var(iou_val))
    print("\nAverage validation accuracy: %f" % np.mean(acc_val))
    print("Variance validation accuracy: %f" % np.var(acc_val))
        
    return np.mean(iou_val), np.mean(acc_val), model