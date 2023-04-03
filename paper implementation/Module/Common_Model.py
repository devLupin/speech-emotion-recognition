import sys
from typing import Tuple
from Config import Config
import numpy as np
from sklearn.metrics import accuracy_score
import prettytable as pt

class Common_Model(object):

    def __init__(self, save_path: str = '', name: str = 'Not Specified'):
        self.model = None
        self.trained = False

    def train(self, x_train, y_train, x_val, y_val):
        raise NotImplementedError()

    
    def predict(self, samples):
        raise NotImplementedError()
        

    def predict_proba(self, samples):
        if not self.trained:
            sys.stderr.write("No Model.")
            sys.exit(-1)
        return self.model.predict_proba(samples)


    def save_model(self, model_name: str):
        raise NotImplementedError()


    def evaluate(self, x_test, y_test):
        tb = pt.PrettyTable()
        temp = ["###"]
        print(temp)
        for item in Config.CLASS_LABELS:
            temp.append(item)
        temp.append("All")
        temp.append("Correct")
        temp.append("Accuracy")
        tb.field_names = temp
        predictions = self.predict(x_test)
        y_test = np.argmax(y_test,axis=1)
        num = len(y_test)
        
        emotion_num = np.zeros((20, 20), dtype=np.int)
        print(y_test)
        print(predictions)
        for i in range(num):
             emotion_num[y_test[i]][10] += 1
             emotion_num[y_test[i]][predictions[i]] += 1
             if y_test[i]==predictions[i]:
                 emotion_num[y_test[i]][11]+=1
        
        for i in range(len(Config.CLASS_LABELS)):
            print(i,'类acc:',emotion_num[i][11]/emotion_num[i][10])

        print('Accuracy:%.3f\n' % accuracy_score(y_pred = predictions, y_true = y_test))
        
        
        
        for i in range(len(Config.CLASS_LABELS)):
            temp = []
            temp.append(Config.CLASS_LABELS[i])
            for j in range(len(Config.CLASS_LABELS)):
                temp.append(emotion_num[i][j])
            temp.append(emotion_num[i][10])
            temp.append(emotion_num[i][11])
            temp.append(emotion_num[i][11]/emotion_num[i][10])
            tb.add_row(temp)

        print(tb)
        '''
        predictions = self.predict(x_test)
        score = self.model.score(x_test, y_test)
        print("True Lable: ", y_test)
        print("Predict Lable: ", predictions)
        print("Score: ", score)
        '''