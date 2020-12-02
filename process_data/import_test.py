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
    image = cv2.imread(name)
    test = np.asarray(cv2.resize(image,dsize=(250,250), interpolation=cv2.INTER_CUBIC))
    test = test[:,:,[2,1,0]]
    plt.imshow(test)
    test = torch.tensor(np.transpose(test))
    test = test.float()
    plt.show()


    ypred = torch.squeeze(model(torch.unsqueeze(test,0).cuda())).cpu().detach().numpy()
    plt.imshow(np.transpose(np.around(ypred)))