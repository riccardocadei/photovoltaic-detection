from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, dataset
from loss.loss import *
import numpy as np
import time
from torch.autograd import Variable
import torch

def training_model(train_loader,loss_function,optimizer,model,num_epochs,scheduler=None):

    if scheduler == None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=1, last_epoch=-1, verbose=False)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
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
            
            running_loss += loss.item()
        scheduler.step()
        if (epoch % 10) == 0:
            print('Epoch n.',epoch, 'Loss',np.around(running_loss/len(train_loader),4),'Time Remaining',np.around((num_epochs-epoch)*(time.time()-t0)/60,4))
    return model


def test_model(test_loader,optimizer,model):
    
    iou_test = []
    acc_test = []
    for i, (images,labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
        prediction = model(images)
        iou_i = iou(np.around(prediction.detach().cpu().numpy()),labels.detach().cpu().numpy())
        iou_test.append(iou_i)
        acc_i = accuracy(np.around(prediction.detach().cpu().numpy()),labels.detach().cpu().numpy())
        acc_test.append(acc_i)
        
    return np.mean(iou_test), np.mean(acc_test)




def cross_validation(train_dataset,loss_function,input_model,num_epochs,lr):
    
    iou_test = []
    acc_test = []
    #define kfold
    kfold =KFold(n_splits=5,shuffle=True)
    for fold, (train_index, test_index) in enumerate(kfold.split(train_dataset)): 
        # split into k Folders
        train_fold = dataset.Subset(train_dataset,train_index)
        test_fold = dataset.Subset(train_dataset,test_index) 
        train_fold_loader = DataLoader(train_fold,batch_size=2, shuffle=True,num_workers=2)
        test_fold_loader = DataLoader(test_fold,batch_size=2, shuffle=True,num_workers=2)
        
        #train the model
        optimizer = torch.optim.SGD(input_model.parameters(), lr=lr)
        model = training_model(train_fold_loader,loss_function,optimizer,input_model,num_epochs)
        # make prediction and compute the evaluation metrics
        iou, acc = test_model(test_fold_loader,optimizer,model)
        print('Iter {}: IoU = {:.4} /  Accuracy = {:.4}'.format(fold, iou, acc))
        iou_test.append(iou)
        acc_test.append(acc)
        
    print("\nAverage test IoU: %f" % np.mean(iou_test))
    print("Variance test IoU: %f" % np.var(iou_test))
    print("\nAverage test accuracy: %f" % np.mean(acc_test))
    print("Variance test accuracy: %f" % np.var(acc_test))
        
    return np.mean(iou_test), np.mean(acc_test), model


def select_hyper_param(train_dataset,loss_function,input_model,num_epochs,lr_candidates):
    
    comparison = []
    for lr in lr_candidates:
        print('---------------------------------------------------------------------\n')
        print('Learning Rate = {}\n'.format(lr))
        iou, acc, model = cross_validation(train_dataset, loss_function, input_model, num_epochs, lr)
        comparison.append([lr, iou, acc, model])
    comparison = np.array(comparison).reshape(len(lr_candidates),4)
    ind_best =  np.argmax(comparison[:,1]) 
    best_lr = comparison[ind_best,0]
    best_model = comparison[ind_best,3]
    best_iou = np.max(comparison[:,1])
        
    return best_lr, best_model, best_iou