from scipy.io import loadmat
import numpy as np
from catboost import *
from catboost import metrics
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

class CatBoost:
    def __init__(self, data):
        self.X_train = self.get_X(data, 1, 2)
        self.y_train = self.get_y(data, 1, 2)
        self.X_test = self.get_X(data, 0, 1)
        self.y_test = self.get_y(data, 0, 1)
        self.model = CatBoostClassifier(iterations=50, verbose=5, random_seed=42, learning_rate=0.1, custom_loss=[metrics.Accuracy(), metrics.F1()])

    def get_X(self, data, l, r):
        matrices = [csr_matrix(data[f'Day{i}']['data'][0][0]) for i in range(l, r)]
        X = vstack(matrices, format='csr')

        return X

    def get_y(self, data, l, r):
        y = []
        for i in range(l, r):
            day_i_labels = data[f'Day{i}']['labels'][0][0]
            for j in range(0, len(day_i_labels)):
                y += day_i_labels[j].tolist()

        return y

    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        return self.model.predict(self.X_test)
  
def main():
    Data = loadmat('url.mat')
    
    cb_clf = CatBoost(Data)
    cb_clf.fit()
    y_predict = cb_clf.predict()

    acc_score = accuracy_score(cb_clf.y_test, y_predict)
    tn, fp, fn, tp = confusion_matrix(cb_clf.y_test, y_predict).ravel().tolist()
    f1 = f1_score(cb_clf.y_test, y_predict)

    with open("catboost.txt", "w") as f:
        y_len = len(cb_clf.y_test)
        f.write(f"accuracy: {acc_score}%\nf1 score: {f1}\ntn: {tn} ({tn/y_len*100}%)\nfp: {fp} ({fp/y_len*100}%)\nfn: {fn} ({fn/y_len*100}%)\ntp: {tp} ({tp/y_len*100}%)\n")


if __name__=="__main__":
    main()