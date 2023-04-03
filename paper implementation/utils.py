import random
import numpy as np
import os
import tensorflow as tf

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    
# Smooth label operation
def smooth_labels(labels, factor=0.1):
    """
        smooth the labels
        returned the smoothed labels
    """
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels

#### https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal

def get_metric_calc(conf_matrix):
    """Weighted Average Recall(WAR),
       Unweighted Average Recall(UAR)

    Args:
        conf_matrix (numpy array)
    
    return:
        WAR, UAR value
    """
    
    sz = conf_matrix.shape[0]
    
    all_elem = 0
    TP_li = []
    FN_li = []
    speeches_li = []
    for i in range(sz):
        TP = 0
        FN = 0
        num_speeches = 0
        for j in range(sz):
            if i==j:
                TP += conf_matrix[i][j]
            else:
                FN += conf_matrix[i][j]
            all_elem += 1
            num_speeches += 1
        
        TP_li.append(TP)
        FN_li.append(FN)
        speeches_li.append(num_speeches)
    
    WAR = 0.
    for i in range(sz):
        a = (speeches_li[i] / all_elem)
        b = TP_li[i] / (TP_li[i] + FN_li[i])
        WAR = WAR + a*b
    WAR *= 100    
        
    UAR = 0.
    for i in range(sz):
        temp = TP_li[i] / (TP_li[i] + FN_li[i])
        UAR += temp
    UAR = 1/sz * UAR
    UAR *= 100
    
    print(f'WAR(Weighted Average Recall)   : {WAR:.2f}%')
    print(f'UAR(Unweighted Average Recall) : {UAR:.2f}%')