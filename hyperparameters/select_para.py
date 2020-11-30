from loss.loss import *
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, dataset
from loss.loss import *

def training_model(train_loader,loss_function,optimizer,model,num_epochs=10):
    for epoch in range(num_epochs):
        print(epoch)
        for i, (images,labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images=Variable(images.cuda())
                labels=Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(torch.squeeze(outputs), torch.squeeze(labels))
            loss.backward()
            optimizer.step()
    return model

def test_model(test_loader,optimizer,model,num_epochs=10):
    iou_acu = []
    for epoch in range(num_epochs):
        print(epoch)
        for i, (images,labels) in enumerate(test_loader):
            if torch.cuda.is_available():
                images=Variable(images.cuda())
                labels=Variable(labels.cuda())

        prediction = model(images)
        iou_test = iou(prediction.detach().numpy(),labels.detach().numpy())
        iou_auc += [iou_test]
    return mean(iou_acu)


def selec_hyper_para(train_dataset,loss_function,input_model,num_epochs,lr_candidates):
    comparison = []
    #define kfold
    kfold =KFold(n_splits=5,shuffle=True)
    for fold,(train_index, test_index) in enumerate(kfold.split(train_dataset)):  
        print(train_index, test_index)
        # split into k Folders
        train_fold = dataset.Subset(train_dataset,train_index)
        test_fold = dataset.Subset(train_dataset,test_index) 
        train_fold_loader = DataLoader(train_fold,batch_size=2, shuffle=True,num_workers=2)
        test_fold_loader = DataLoader(test_fold,batch_size=2, shuffle=True,num_workers=2)
        
        #train the model 
        for lr in lr_candidates:
            optimizer = torch.optim.SGD(input_model.parameters(), lr=lr)
            model = training_model(train_fold_loader,loss_function,optimizer,input_model,num_epochs)
            # make prediction and compute loss
            loss = test_model(test_fold_loader,optimizer,model,num_epochs)
            comparison += [lr,loss]
    return comparison