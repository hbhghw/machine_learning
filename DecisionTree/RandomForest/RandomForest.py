from tree import Tree
import random


class RandomForest:
    def __init__(self, n_tree=100, n_feature=0.7, max_depth=100):
        self.n_tree = n_tree
        self.n_feature = n_feature
        self.max_depth = max_depth
        self.trees = []

    def getData(self, samples):
        #random data from samples for each tree
        ret = []
        total = len(samples)
        for i in range(total):
            ret.append(samples[random.randint(0, total - 1)])
        return ret

    def getIndices(self, indices):
        #random features(indices) for each tree
        if isinstance(self.n_feature, float):
            n = int(self.n_feature * len(indices))
        else:
            n = min(self.n_feature, len(indices))
        ret = indices.copy()
        for i in range(len(indices) - n):
            ret.pop(random.randint(0, len(ret) - 1))
        return ret

    def train(self, samples, indices=None):
        if indices is None:
            indices = list(range(len(samples[0]) - 1))
        for i in range(self.n_tree):
            _samples = self.getData(samples)
            _indices = self.getIndices(indices)
            tree = Tree(max_depth=self.max_depth)
            tree.train(_samples, _indices)
            self.trees.append(tree)

    def predict(self, sample):
        result = {}
        maxcout = 0
        retlabel = None
        for tree in self.trees:
            label = tree.predict(sample)
            if label in result.keys():
                result[label] += 1
            else:
                result[label] = 1
            if result[label] > maxcout:
                maxcout = result[label]
                retlabel = label
        return retlabel


if __name__ == "__main__":
    rf = RandomForest(n_tree=201)
    from readData import getData

    data, labels = getData()
    rf.train(data)
    correct = 0
    total = len(data)
    for i in range(len(data)):
        label = rf.predict(data[i][:-1])
        if data[i][-1] == label:
            correct += 1
    print(correct, total, correct / total)
