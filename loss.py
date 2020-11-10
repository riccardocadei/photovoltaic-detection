import numpy as np
from sklearn.metrics import *

def iou(pred,target):
    '''Compute the intersection over union between a prediction and a targer'''
    intersection = np.logical_and(pred, target)
    union = np.logical_or(target, target)
    
    return np.sum(intersection)/np.sum(union)

def accuracy(pred,target,normalize=True, sample_weight=None):
    '''Compute the accuracy between a prediction and a target, which is equivalent to (TP+TN)/total'''
    return np.mean(pred==target)

def recall(pred,target):
    '''Compute the recall between a prediction and a target, which is equivalent to (TP)/(TP+FN)'''
    TP = np.sum(np.logical_and(pred == 1, target == 1))
    FN = np.sum(np.logical_and(pred == 0, target == 1))
    return TP/(TP + FN)
