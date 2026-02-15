from scipy.io import loadmat
import numpy as np
import scipy.sparse
from scipy.sparse import vstack
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import time

class NaiveBayes:
    def __init__(self, data):
        self.real_features = np.array([i-1 for i in self.get_real_features(data['FeatureTypes'])])
        self.num_of_features = data['Day0']['data'][0][0].shape[1]
        self.X_train_real, self.X_train_binary = self.get_X(data, 0, 62)
        self.Y_train = self.get_Y(data, 0, 62)
        self.X_test_real, self.X_test_binary = self.get_X(data, 62, 121)
        self.Y_test = self.get_Y(data, 62, 121)
        self.gnb_clf = GaussianNB()
        self.bnb_clf = BernoulliNB()
        #self.X_train, self.X_test, self.y_train, self.y_test = self.split_train_test()
        
    def get_X(self, data, l, r):
        matrices = [csr_matrix(data[f'Day{i}']['data'][0][0]) for i in range(l, r)]
        X = vstack(matrices, format='csr')

        columns = np.arange(self.num_of_features)
        binary_features_indices = np.setdiff1d(columns, self.real_features)
        
        return X[:, self.real_features].toarray(), X[:, binary_features_indices]
        

    def get_Y(self, data, l, r):
        y = []
        for i in range(l, r):
            day_i_labels = data[f'Day{i}']['labels'][0][0]
            for j in range(0, len(day_i_labels)):
                y += day_i_labels[j].tolist()

        return y

    def get_real_features(self, real_features):
        new_real_features = []
        for elem in real_features:
            new_real_features += elem.tolist()

        return new_real_features
    
    def split_train_test(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def fit(self):
        self.gnb_clf.fit(self.X_train_real, self.Y_train)
        self.bnb_clf.fit(self.X_train_binary, self.Y_train)

    def predict(self):
        log_prob_gauss = self.gnb_clf.predict_log_proba(self.X_test_real)
        log_prob_bernoulli = self.bnb_clf.predict_log_proba(self.X_test_binary)
        log_prob = log_prob_gauss + log_prob_bernoulli
        
        return np.argmax(log_prob, axis=1)

def main():
    data = loadmat('url.mat')
    
    model = NaiveBayes(data)
    model.fit()
    y_predict = model.predict()
    
    accuracy = accuracy_score(model.Y_test, y_predict)
    f1score = f1_score(model.Y_test, y_predict)
    tn, fp, fn, tp = confusion_matrix(model.Y_test, y_predict).ravel().tolist()
    
    with open("naive_bayes.txt", "w") as f:
        y_len = len(model.Y_test)
        f.write(f"accuracy: {accuracy}%\nf1 score: {f1score}\ntn: {tn} ({tn/y_len*100}%)\nfp: {fp} ({fp/y_len*100}%)\nfn: {fn} ({fn/y_len*100}%)\ntp: {tp} ({tp/y_len*100}%)\n")

if __name__=="__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))