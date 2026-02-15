from scipy.io import loadmat
import numpy as np
import scipy.sparse
from scipy.sparse import vstack
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

class LR:
    def __init__(self, data):
        self.X_train = self.get_X(data, 1, 2)
        self.Y_train = self.get_Y(data, 1, 2)
        self.X_test = self.get_X(data, 0, 1)
        self.Y_test = self.get_Y(data, 0, 1)
        self.num_of_features = data['Day0']['data'][0][0].shape[1]
        
    def get_X(self, data, l, r):
        matrices = [data[f'Day{i}']['data'][0][0] for i in range(l, r)]
        X = vstack(matrices, format='csc')
    
        return X

    def get_Y(self, data, l, r):
        y = []
        for i in range(l, r):
            day_i_labels = data[f'Day{i}']['labels'][0][0]
            for j in range(0, len(day_i_labels)):
                y += day_i_labels[j].tolist()

        return y

    def split_train_test(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.1, random_state=42)
        return X_train, X_test, y_train, y_test

    def fit(self):
        model = LogisticRegression(max_iter=1000, solver='saga', random_state=42)
        model.fit(self.X_train, self.Y_train)
        
        return model
    
def main():
    Data = loadmat('url.mat')

    lr_clf = LR(Data)
    model = lr_clf.fit()
    y_test_pred = model.predict(lr_clf.X_test)


    accuracy = accuracy_score(lr_clf.Y_test, y_test_pred)
    f1score = f1_score(lr_clf.Y_test, y_test_pred)
    tn, fp, fn, tp = confusion_matrix(lr_clf.Y_test, y_test_pred).ravel().tolist()

    with open("logistic_regression.txt", "w") as f:
        y_len = len(lr_clf.Y_test)
        f.write(f"accuracy: {accuracy}%\nf1 score: {f1score}\ntn: {tn} ({tn/y_len*100}%)\nfp: {fp} ({fp/y_len*100}%)\nfn: {fn} ({fn/y_len*100}%)\ntp: {tp} ({tp/y_len*100}%)\n")


if __name__=="__main__":
    main()
