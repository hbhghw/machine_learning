import numpy as np
from stump import Stump
import pickle
import sys

class AdaboostClassifier:
    def __init__(self,n_weakClassifier=10):
        self.n_weakClassifier = n_weakClassifier
        self.stumps = []

    def initWeights(self,samples):
        return  [1/samples.shape[0] for _ in range(samples.shape[0])]

    def train(self,samples,weights=None):
        print('----------------start training-----------------')
        if weights is None:
            weights = self.initWeights(samples)
        stump0 = Stump()
        stump0.train(samples,weights)
        self.stumps.append(stump0)
        for _ in range(1,self.n_weakClassifier):
            weights = self.stumps[-1].updateWeights
            stump_i = Stump()
            stump_i.train(samples,weights)
            self.stumps.append(stump_i)
        print("----------adaboost training finished.-----------")

    def predict(self,sample):
        score = 0
        for stump in self.stumps:
            score += stump.predict(sample)
        return score

if __name__ == '__main__':
    samples = np.load('data.npy')
    if sys.argv[1]=='train':
        ada = AdaboostClassifier(20)
        ada.train(samples)
        correct = 0
        for sample in samples:
            score = ada.predict(sample)
            if score * sample[-1] > 0:
                correct += 1
        print('Weak classifiers number :',ada.n_weakClassifier)
        print('Adaboost testing accuracy :', correct / samples.shape[0])
        with open('model','wb') as f:
            pickle.dump(ada, f)
    else:
        with open('model','rb') as f:
            ada = pickle.load(f)
        correct = 0
        for sample in samples:
            score = ada.predict(sample)
            if score * sample[-1] > 0:
                correct += 1
        print('Adaboost testing accuracy:', correct / samples.shape[0])