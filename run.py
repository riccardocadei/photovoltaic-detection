
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from model.unet import *
from loss.loss import *
from process_data.data_loader import *


if __name__ ==  '__main__':
  

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # %% [markdown]
    # # Loading the Data Set
    # First we load the data set that we will use for training. Each sample is an image with its mask (label). An image is represented as a 3x250x250 array with each of the 3 color chanel being 250x250 pixels. The asssociated mask is a 250x250 array, 

    # %%
    folder_path_image = 'data/image'
    folder_path_mask  = 'data/mask'

    train_dataset = DataLoaderSegmentation(folder_path_image,folder_path_mask)
    train_loader = DataLoader(train_dataset,batch_size=5, shuffle=True,num_workers=2)

    # %% [markdown]
    # # Augment the data set
    # We can build a more diverse and robust training set by applying transformations to the manually labeled images
    # %% [markdown]
    # # Initiate the model
    # In this report, we will use the Unet model presented in medical image segmentation, and in the previous papers of the Professor.

    # %%
    model = UNet(3,1,False).to(device)
    print(model)

    # %% [markdown]
    # # Loss & Optimizer

    # %%
    loss_function = torch.nn.BCEWithLogitsLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # %% [markdown]
    # # Training Loop

    # %%
    num_epochs = int(input("Number of epochs:"))


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
        print(loss)

    torch.save(model.state_dict(), 'model/trained_model.pt')