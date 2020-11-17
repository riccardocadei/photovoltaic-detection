import numpy as np

def iou(pred,target):
    '''Compute the intersection over union between a prediction and a targer
    
    Args:
        pred (np.array): predicted value
        target (np.array): targeted value, usually the label

    Returns:
        double: precision between pred and target
    '''
    intersection = np.logical_and(pred, target)
    union = np.logical_or(pred, target)
    
    return np.sum(intersection)/np.sum(union)

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