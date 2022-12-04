# -*- coding:UTF-8 -*-
# Library import
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from Utils import get_all_data
from Models import CPAC_Model, LIGHT_SERNET
import pandas as pd
from Config import Config
import numpy as np

# GPU configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)

# Paths and parameter settings
DATA_PATH = 'RAVDESS-deltas'
CLASS_LABELS = Config.CLASS_LABELS
model_name = 'LIGHT_SERNET'
feature_name = 'mfcc'
epoch = 300
k = 10
random_seed = 98
repeat_number = 1

# Read data
with open(f'dataset/{DATA_PATH}.npy', 'rb') as f:
    x = np.load(f)
    y = np.load(f)
    
# Create model
y = to_categorical(y,num_classes=len(Config.CLASS_LABELS))
data_shape = x.shape[1:]
model = LIGHT_SERNET(input_shape = data_shape, num_classes = len(Config.CLASS_LABELS))

print(x.shape, data_shape)

# Train model
print('Start Train CPAC in Single-Corpus')
for number_times in range(0,repeat_number):
    i = 0
    data1 = []
    df = []
    model.matrix = []
    
    model.train(x, y, None, None, n_epochs=epoch, data_name=DATA_PATH, fold=k, random=random_seed+number_times, model_name=model_name, feature_name=feature_name)
    
    naming = f'Results/{DATA_PATH}/{model_name}_{feature_name}_{k}-fold'
    naming = f'{naming}_{str(round(model.acc*10000)/100)}.xlsx'
    
    writer = pd.ExcelWriter(naming)
    for i,item in enumerate(model.matrix):
        temp = {}
        temp[" "] = CLASS_LABELS
        j = 0
        for j,l in enumerate(item):
            temp[CLASS_LABELS[j]]=item[j]
        data1 = pd.DataFrame(temp)
        data1.to_excel(writer,sheet_name=str(i), encoding='utf8')

        df = pd.DataFrame(model.eva_matrix[i]).transpose()
        df.to_excel(writer,sheet_name=str(i)+"_evaluate", encoding='utf8')

    writer.save()
    writer.close()
    tf.keras.backend.clear_session()
    print('End of the '+str(number_times+1)+' training session')
