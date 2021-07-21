import numpy as np
import math

import torch.nn.functional as F
import torch.nn as nn
import torch
from scipy.ndimage import distance_transform_edt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def iou(pred,target):
    '''Compute the intersection over union between a prediction and a targer
    
    Args:
        pred (np.array): predicted value
        target (np.array): targeted value, usually the label

    Returns:
        double: precision between pred and target
    '''
    if np.all(target == 0):
        target = 1 - target
        pred = 1-pred

    intersection = np.logical_and(pred, target)
    union = np.logical_or(pred, target)
    result = np.sum(intersection)/np.sum(union)
    if math.isnan(result):
        result = 0
    
    return result

def accuracy(pred,target,normalize=True, sample_weight=None):
    '''Compute the accuracy between a prediction and a target, which is equivalent to (TP+TN)/total
    
    Args:
        pred (np.array): predicted value
        target (np.array): targeted value, usually the label

    Returns:
        double: accuracy between pred and target
    '''
    return np.mean(pred==target)

def recall(pred,target):
    """Compute the recall between a prediction and a target, which is equivalent to (TP)/(TP+FN)

    Args:
        pred (np.array): predicted value
        target (np.array): targeted value, usually the label

    Returns:
        double: Recall between pred and target
    """
    TP = np.sum(np.logical_and(pred == 1, target == 1))
    FN = np.sum(np.logical_and(pred == 0, target == 1))
    return TP/(TP + FN)

def precision(pred,target):
    TP = np.sum(np.logical_and(pred == 1, target == 1))
    FP = np.sum(np.logical_and(pred == 1, target == 0))
    if (TP + FP) == 0:
        return 0
    else:
        return TP/(TP + FP)