import numpy as np
import matplotlib.pyplot  as plt
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader

from model.unet import *
from loss.loss import *
from process_data.data_loader import *
from hyperparameters.select_param import *
from matplotlib import image
from PIL import Image
import cv2



def import_and_show(model,name):
    fig = plt.figure()
    fig.set_size_inches(12, 7, forward=True)

    ax1 = fig.add_subplot(1,2,1)
    ax1.title.set_text('Input Image')
    ax2 = fig.add_subplot(1,2,2)
    ax2.title.set_text('Predicted Label')
    image = cv2.imread(name)
    test = np.asarray(cv2.resize(image,dsize=(250,250), interpolation=cv2.INTER_CUBIC))
    test = test[:,:,[2,1,0]]
    ax1.imshow(test)
    test = torch.tensor(np.transpose(test))
    test = test.float()

    ypred = torch.squeeze(model.predict(torch.unsqueeze(test,0).cuda())).cpu().detach().numpy()
    ax2.imshow(np.transpose(np.around(ypred)))
    plt.show()