import numpy as np
from tree import Tree

class AdaboostClassifier:
    def __init__(self,n_weakClassifier=10):
        self.n_weakClassifier = n_weakClassifier
        self.trees = []

    def initWeights(self,samples):
        return  [1/samples.shape[0] for _ in range(samples.shape[0])]

    def train(self,samples,weights=None):
        if weights is None:
            weights = self.initWeights(samples)
        tree0 = Tree()
        tree0.train(samples,weights)
        self.trees.append(tree0)
        for _ in range(1,self.n_weakClassifier):
            weights = self.trees[-1].updateWeights
            tree_i = Tree()
            tree_i.train(samples,weights)
            self.trees.append(tree_i)
        print("Adaboost training finished.")

    def predict(self,sample):
        score = 0
        for tree in self.trees:
            score += tree.predict(sample)
        return score

if __name__ == '__main__':
    samples = np.load('data.npy')
    ada = AdaboostClassifier(10)
    ada.train(samples)
    correct = 0
    for sample in samples:
        score = ada.predict(sample)
        if score*sample[-1]>0:
            correct += 1
    print('acc:',correct/samples.shape[0])