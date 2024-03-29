import numpy as np
import sys
import os

import tensorflow as tf
import tensorflow.keras.layers
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers,callbacks
from tensorflow.keras.layers import Layer, Lambda, Conv2D, Dropout,Dense,Activation,Input,GlobalAveragePooling1D, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.layers import Reshape,Flatten,BatchNormalization,MaxPooling1D,AveragePooling2D,Reshape,Attention, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping,History,ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from Common_Model import Common_Model
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,f1_score
from datetime import datetime
from Config import Config

# Random seed setting
from numpy.random import seed
seed(1024)
tf.random.set_seed(2048)

def LearningRateScheduler():
    def scheduler(epoch, lr):
        LEARNING_RATE_DECAY_PARAMETERS = -0.15
        LEARNING_RATE_DECAY_STRATPOINT = 50
        LEARNING_RATE_DECAY_STEP = 20
        
        if epoch < LEARNING_RATE_DECAY_STRATPOINT:
            return lr
        else:
            if epoch % LEARNING_RATE_DECAY_STEP == 0:
                lr = lr * tf.math.exp(LEARNING_RATE_DECAY_PARAMETERS)
        return lr
    return tf.keras.callbacks.LearningRateScheduler(scheduler)


def margin_loss(y_true, y_pred):
    """
        Margin Loss
        :param y_true: [None, n_classes]
        :param y_pred: [None, num_capsule]
        :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))



def softmax(x, axis=-1):
    """
        softmax in Dynamic Routings
    """
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)

def PrimaryCapssquash(vectors, axis=-1):
    """
        The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
        :param vectors: some vectors to be squashed, N-dim tensor
        :param axis: the axis to squash
        :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

def DigitCapssquash(Value, axis = -1):
    """
        Squash activation in PrimaryCaps
        :return: a Tensor with same shape as input vectors
    """
    Square_Vector = K.sum(K.square(Value), axis, keepdims=True)
    Proportion = Square_Vector / (1 + Square_Vector) / K.sqrt(Square_Vector + K.epsilon())
    Output = Proportion * Value
    return Output

def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
        Apply Conv2D `n_channels` times and concatenate all capsules
        :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    output = tf.keras.layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    outputs = tf.keras.layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return tf.keras.layers.Lambda(PrimaryCapssquash, name='primarycap_squash')(outputs)

# Smooth label operation
def smooth_labels(labels, factor=0.1):
    """
        smooth the labels
        returned the smoothed labels
    """
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels

class Capsule(tensorflow.keras.layers.Layer):
    """
        DigitCaps layer
    """
    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=True, activation='squash', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = DigitCapssquash
        else:
            self.activation = activations.get(activation)
    def get_config(self):
       config = {"num_capsule":self.num_capsule,
                 "dim_capsule":self.dim_capsule,
                 "routings":self.routings,
                 "share_weights":self.share_weights,
                 "activation":self.activation
                }
       base_config = super(Capsule, self).get_config()
       return dict(list(base_config.items()) + list(config.items()))
    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        #input_dim_capsule = 8
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,#input_dim_capsule = 16
                                            self.num_capsule * self.dim_capsule), #16*32
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))

        b = K.zeros_like(u_hat_vecs[:,:,:,0])

        for i in range(self.routings): #Routings
            c = softmax(b, 1)

            o = tf.einsum('bin,binj->bij', c, u_hat_vecs)
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.routings - 1:
                o = K.l2_normalize(o, -1)
                b = tf.einsum('bij,binj->bin', o, u_hat_vecs)
                if K.backend() == 'theano':
                    b = K.sum(b, axis=1)

        return self.activation(o)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


class CPAC_Model(Common_Model):
    def __init__(self, input_shape, num_classes, **params):
        super(CPAC_Model,self).__init__(**params)
        self.data_shape = input_shape
        self.num_classes = num_classes
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0


    """
        Store the model weights as model_name.h5 and model_name.json in the /Models directory
    """
    # save_model(data_name, model_name, feature_name, fold)
    def save_model(self, dataset_name, model_name, feature_name, fold):
        save_path = os.path.join('Models', dataset_name)
        os.makedirs(save_path, exist_ok=True)
        
        now_time = datetime.now().strftime("%m-%d-%H%M%S")
        
        naming = f'{model_name}_{feature_name}_{fold}-fold_{now_time}'
        
        h5_path = f'{naming}.h5'
        self.model.save_weights(os.path.join(save_path, h5_path))

        json_path = f'{naming}.json'
        with open(os.path.join(save_path, json_path), "w") as json_file:
            json_file.write(self.model.to_json())
    """
        train(): train the model on the given training set
        input:
            x (numpy.ndarray): the training set samples
            y (numpy.ndarray): training set labels
            x_val (numpy.ndarray): test set samples
            y_val (numpy.ndarray): test set labels
            n_epochs (int): number of epochs
    """
    def create_model(self):
        self.inputs=Input(shape=(self.data_shape[0], self.data_shape[1], self.data_shape[2]))
        self.conv_1 = Conv2D(filters=64, kernel_size=3, name=None)(self.inputs)
        self.conv_1 = BatchNormalization(axis=-1)(self.conv_1, training = False)
        self.conv_1 = Activation('elu')(self.conv_1)
        self.conv_1 = AveragePooling2D()(self.conv_1)
        self.conv_1 = Dropout(0.5)(self.conv_1)
        
        self.conv_2 = Conv2D(filters=64, kernel_size=3, name=None)(self.conv_1)
        self.conv_2 = BatchNormalization(axis=-1)(self.conv_2, training = False)
        self.conv_2 = Activation('elu')(self.conv_2)
        self.conv_2 = AveragePooling2D()(self.conv_2)
        self.conv_2 = Dropout(0.25)(self.conv_2)

        self.conv_3 = Conv2D(filters=64, kernel_size=3, name=None)(self.conv_2)
        self.conv_3 = BatchNormalization(axis=-1)(self.conv_3, training = False)
        self.conv_3 = Activation('elu')(self.conv_3)
        self.conv_3 = AveragePooling2D()(self.conv_3)
        self.conv_3 = Dropout(0.25)(self.conv_3)
        self.cap = self.conv_3
        self.primarycaps = PrimaryCap(self.cap, dim_capsule=64, n_channels=6, kernel_size=3, strides=1,padding='valid')


        self.cap = self.primarycaps
        self.sa = Attention(use_scale =True )([self.primarycaps,self.primarycaps,self.primarycaps])
        self.cap = Lambda(lambda x: tf.multiply(x[0], x[1]))([self.cap, self.sa])

        self.capsule = Capsule(6,64,3,True)(self.cap)

        self.GA = GlobalAveragePooling1D()(self.capsule)
        self.drop = Dropout(0.2)(self.GA)
        self.predictions = Dense(self.num_classes,activation='softmax')(self.drop)
        self.model = Model(inputs = self.inputs,outputs = self.predictions)
        self.model.compile(loss = [margin_loss, 'mse'],
                           loss_weights=[1., 0.392],
                           optimizer = Adam(learning_rate=0.001,beta_1=0.975, beta_2=0.932,epsilon=1e-8), 
                           metrics = ['accuracy'])
        print("Model create succes!")
        print(self.model.summary())
    
    def train(self, x, y, x_test = None, y_test =None, n_epochs = None, data_name=None, fold = None , random = None, model_name=None, feature_name=None):
        emotions_groundtruth_list = np.array([])
        predicted_emotions_list = np.array([])
        
        avg_accuracy = 0
        avg_loss = 0
        n_split = fold
        filepath='./Models/'
        kfold = KFold(n_splits=n_split, shuffle=False, random_state= random)
        i=1
        for train, test in kfold.split(x, y):
            self.create_model()
            y[train] = smooth_labels(y[train], 0.1)
            folder_address = filepath+data_name
            if not os.path.exists(folder_address):
                os.mkdir(folder_address)
            max_acc = 0
            max_f1 = 0
            best_eva_list = []
            for epoch in range(n_epochs):
                print("epoch/max_epochs:",str(epoch+1)+'/'+str(n_epochs)+" best_acc:"+str(round(max_acc*10000)/100)+" best_F1:"+str(round(max_f1*10000)/100))
                self.model.fit(x[train], y[train],batch_size = 64,epochs = 1,verbose=1)
                evaluate_list = self.model.evaluate(x[test],  y[test])
                y_pred = self.model.predict(x[test])
                f1 = f1_score(np.argmax(y[test],axis=1), np.argmax(y_pred,axis=1), average='weighted')
                if evaluate_list[1]>max_acc:
                    max_acc = evaluate_list[1]
                    best_eva_list = evaluate_list
                    y_pred_best = y_pred
                if f1>max_f1:
                    max_f1 = f1

            avg_loss += best_eva_list[0]
            avg_accuracy += best_eva_list[1]
            print(str(i)+'_Model evaluation: ', evaluate_list,"   Now ACC:",str(round(avg_accuracy*10000)/100/i))
            i+=1
            self.matrix.append(confusion_matrix(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1)))

            em = classification_report(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1), target_names=Config.CLASS_LABELS,output_dict=True)
            self.eva_matrix.append(em)
            print(classification_report(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1), target_names=Config.CLASS_LABELS))

            emotions_groundtruth_list = np.append(emotions_groundtruth_list, np.argmax(y[test],axis=1))
            predicted_emotions_list = np.append(predicted_emotions_list, np.argmax(y_pred_best,axis=1))


        Report = classification_report(emotions_groundtruth_list, predicted_emotions_list)
        
        os.makedirs(f'Results/{data_name}', exist_ok=True)
        
        report_path = f'Results/{data_name}/{model_name}_{feature_name}_{fold}-fold_nomalize.txt'
        with open(report_path, "w") as f:
            f.write(Report)


        print("Average ACC:",avg_accuracy/n_split)
        self.acc = avg_accuracy/n_split
        self.trained = True
        
        self.save_model(data_name, model_name, feature_name, fold)

    """
        predict(): identify the emotion of the audio
        input:
            samples: the audio features to be recognized
        Output:
            list: the results
    """
    def predict(self, sample):
        if not self.trained:
            sys.stderr.write("No Model.")
            sys.exit(-1)
        return np.argmax(self.model.predict(sample,verbose=2), axis=1)


class LIGHT_SERNET(Common_Model):

    def __init__(self, input_shape, num_classes, **params):
        super(LIGHT_SERNET,self).__init__(**params)
        self.data_shape = input_shape
        self.num_classes = num_classes
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0

    def save_model(self, dataset_name, model_name, feature_name, fold):
        save_path = os.path.join('Models', dataset_name)
        os.makedirs(save_path, exist_ok=True)
        
        now_time = datetime.now().strftime("%m-%d-%H%M%S")
        
        naming = f'{model_name}_{feature_name}_{fold}-fold_{now_time}'
        
        h5_path = f'{naming}.h5'
        self.model.save_weights(os.path.join(save_path, h5_path))

        json_path = f'{naming}.json'
        with open(os.path.join(save_path, json_path), "w") as json_file:
            json_file.write(self.model.to_json())
            
    def create_model(self):
        self.inputs=Input(shape=(self.data_shape[0], self.data_shape[1], self.data_shape[2]))
        
        self.path1 = Conv2D(filters=32, kernel_size=(11,1), padding='same', strides=(1,1))(self.inputs)
        self.path2 = Conv2D(filters=32, kernel_size=(1,9), padding='same', strides=(1,1))(self.inputs)
        self.path3 = Conv2D(filters=32, kernel_size=(3,3), padding='same', strides=(1,1))(self.inputs)
        
        self.path1 = BatchNormalization(axis=-1)(self.path1)
        self.path2 = BatchNormalization(axis=-1)(self.path2)
        self.path3 = BatchNormalization(axis=-1)(self.path3)
        
        self.path1 = ReLU()(self.path1)
        self.path2 = ReLU()(self.path2)
        self.path3 = ReLU()(self.path3)
        
        self.path1 = AveragePooling2D(pool_size=2, padding='same')(self.path1)
        self.path2 = AveragePooling2D(pool_size=2, padding='same')(self.path2)
        self.path3 = AveragePooling2D(pool_size=2, padding='same')(self.path3)
        
        self.features = Concatenate(axis=-1)([self.path1, self.path2, self.path3])
        
        self.x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-6))(self.features)
        self.x = BatchNormalization()(self.x)
        self.x = ReLU()(self.x)
        self.x = AveragePooling2D(pool_size=(2, 2), padding="same")(self.x)
        
        self.x = Conv2D(96, (3,3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-6))(self.x)
        self.x = BatchNormalization()(self.x)
        self.x = ReLU()(self.x)
        self.x = AveragePooling2D(pool_size=(2,2), padding="same")(self.x)


        self.x = Conv2D(128, (3,3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-6))(self.x)
        self.x = BatchNormalization()(self.x)
        self.x = ReLU()(self.x)
        self.x = AveragePooling2D(pool_size=(2,1) , padding="same")(self.x)

        self.x = Conv2D(160, (3,3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-6))(self.x)
        self.x = BatchNormalization()(self.x)
        self.x = ReLU()(self.x)
        self.x = AveragePooling2D(pool_size=(2,1) , padding="same")(self.x)

        self.x = Conv2D(320, (1,1), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(1e-6))(self.x)
        self.x = BatchNormalization()(self.x)
        self.x = ReLU()(self.x)
        self.x = GlobalAveragePooling2D()(self.x)

        self.drop = Dropout(0.3)(self.x)
        
        self.predictions = Dense(self.num_classes, activation='softmax')(self.drop)
        self.model = Model(inputs = self.inputs,outputs = self.predictions)
        
        self.model.compile(loss = [margin_loss, 'mse'],
                           loss_weights=[1., 0.392],
                           optimizer = Adam(learning_rate=0.001,beta_1=0.975, beta_2=0.932,epsilon=1e-8), 
                           metrics = ['accuracy'])
        
        print("Model create succes!")
        print(self.model.summary())
        
    def train(self, x, y, x_test = None, y_test =None, n_epochs = None, data_name=None, fold = None , random = None, model_name=None, feature_name=None):
        emotions_groundtruth_list = np.array([])
        predicted_emotions_list = np.array([])
        
        avg_accuracy = 0
        avg_loss = 0
        n_split = fold
        filepath='./Models/'
        kfold = KFold(n_splits=n_split, shuffle=False, random_state=None)
        i=1
        for train, test in kfold.split(x, y):
            self.create_model()
            
            # learningrate_scheduler = LearningRateScheduler()

            # return_bestweight = BestModelWeights()
    
            # callbacks = [learningrate_scheduler, return_bestweight]
            
            y[train] = smooth_labels(y[train], 0.1)
            
            folder_address = filepath+data_name
            if not os.path.exists(folder_address):
                os.mkdir(folder_address)
                
            max_acc = 0
            max_f1 = 0
            best_eva_list = []
            for epoch in range(n_epochs):
                print("epoch/max_epochs:",str(epoch+1)+'/'+str(n_epochs)+" best_acc:"+str(round(max_acc*10000)/100)+" best_F1:"+str(round(max_f1*10000)/100))
                
                self.model.fit(x[train], 
                               y[train],
                               batch_size=64,
                               epochs=1,
                               verbose=1)
                
                evaluate_list = self.model.evaluate(x[test],  y[test])
                y_pred = self.model.predict(x[test])
                
                f1 = f1_score(np.argmax(y[test],axis=1), np.argmax(y_pred,axis=1), average='weighted')
                
                if evaluate_list[1]>max_acc:
                    max_acc = evaluate_list[1]
                    best_eva_list = evaluate_list
                    y_pred_best = y_pred
                
                if f1>max_f1:
                    max_f1 = f1

            avg_loss += best_eva_list[0]
            avg_accuracy += best_eva_list[1]
            print(str(i)+'_Model evaluation: ', evaluate_list,"   Now ACC:",str(round(avg_accuracy*10000)/100/i))
            i+=1
            self.matrix.append(confusion_matrix(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1)))

            em = classification_report(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1), target_names=Config.CLASS_LABELS,output_dict=True)
            self.eva_matrix.append(em)
            print(classification_report(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1), target_names=Config.CLASS_LABELS))

            emotions_groundtruth_list = np.append(emotions_groundtruth_list, np.argmax(y[test],axis=1))
            predicted_emotions_list = np.append(predicted_emotions_list, np.argmax(y_pred_best,axis=1))


        Report = classification_report(emotions_groundtruth_list, predicted_emotions_list)
        
        os.makedirs(f'Results/{data_name}', exist_ok=True)
        
        report_path = f'Results/{data_name}/{model_name}_{feature_name}_{fold}-fold_nomalize.txt'
        with open(report_path, "w") as f:
            f.write(Report)


        print("Average ACC:",avg_accuracy/n_split)
        self.acc = avg_accuracy/n_split
        self.trained = True
        
        self.save_model(data_name, model_name, feature_name, fold)

    def predict(self, sample):
        if not self.trained:
            sys.stderr.write("No Model.")
            sys.exit(-1)
        return np.argmax(self.model.predict(sample,verbose=2), axis=1)