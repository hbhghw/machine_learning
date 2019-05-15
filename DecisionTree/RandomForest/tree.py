import math
from collections import defaultdict


class Node:
    def __init__(self):
        self.index = -1 #split index in this node
        self.isLeaf = False #is leaf node or not
        self.label = -1 #final label,available only if isLeaf=True
        self.threshold = 0 #threshold
        self.left = None #leftchild
        self.right = None #rightchild


class Tree:
    def __init__(self, num_classes=2, max_depth=100):
        self.root = Node()
        self.num_classes = num_classes
        self.maxDepth = max_depth

    def calculateInfo(self, left, right):
        #info = -sigma(pi*log2(pi))
        leftsum = sum(left)
        rightsum = sum(right)
        info = 0
        for i in left: #left child
            if i != 0:
                info = info - (i / leftsum * math.log2(i / leftsum))
        for j in right:#right child
            if j != 0:
                info = info - (j / leftsum * math.log2(j / rightsum))
        return info

    def calculateInfoAndThreshold(self, samples, index):
        values = [(sample[index], sample[-1]) for sample in samples] #feature value at index and label
        values.sort(key=lambda item: item[0]) #sort by feature value
        mininfo = 9999
        threshold = 0
        left = {}
        right = {}
        for value in values:
            if value[-1] in right.keys():
                right[value[-1]] += 1
            else:
                left[value[-1]] = 0
                right[value[-1]] = 1

        for value in values:#go through the feature values,and choose one as split threshold
            left[value[1]] += 1
            right[value[1]] -= 1
            info = self.calculateInfo(list(left.values()), list(right.values()))
            if info < mininfo:
                mininfo = info
                threshold = value[0]

        return mininfo, threshold

    def getBestIndex(self, samples, indices):#choose best index and split threshold
        minInfo = 9999
        bestindex = -1
        threshold = 0
        for index in indices:
            info, th = self.calculateInfoAndThreshold(samples, index)
            if info < minInfo:
                minInfo = info
                bestindex = index
                threshold = th
        return bestindex, threshold

    def split(self, samples, index, threshold):#split data at index
        left, right = [], []
        for sample in samples:
            if sample[index] < threshold:
                left.append(sample)
            else:
                right.append(sample)
        return left, right

    def isAllSame(self, samples):#if all label are same,no need to split
        return len(set([sample[-1] for sample in samples])) == 1

    def getMostLabel(self, samples):#find most samples label at this node
        labels = defaultdict(lambda: 0)
        label = None
        maxcount = 0
        for sample in samples:
            labels[sample[-1]] += 1
            if labels[sample[-1]] > maxcount:
                maxcount = labels[sample[-1]]
                label = sample[-1]
        return label

    def train(self, samples, indices=None):
        if indices is None:
            indices = list(range(len(samples[0]) - 1))
        self._train(self.root, samples, indices, 0)

    def _train(self, node, samples, indices, depth):
        if self.isAllSame(samples) or len(indices) == 0:
            node.label = self.getMostLabel(samples)
            node.isLeaf = True
            return
        bestIndex, threshold = self.getBestIndex(samples, indices)
        if bestIndex == -1 or depth > self.maxDepth:
            node.label = self.getMostLabel(samples)
            node.isLeaf = True
            return
        node.index = bestIndex
        node.threshold = threshold
        left_samples, right_samples = self.split(samples, bestIndex, threshold)
        indices.remove(bestIndex)
        #train left child
        lnode = Node()
        self._train(lnode, left_samples, indices, depth + 1)
        node.left = lnode
        #train right child
        rnode = Node()
        self._train(rnode, right_samples, indices, depth + 1)
        node.right = rnode

    def predict(self, sample):
        node = self.root
        while not node.isLeaf:
            if sample[node.index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.label


if __name__ == '__main__':
    from readData import getData

    data, labels = getData()
    tree = Tree()
    tree.train(data)
    correct = 0
    total = len(data)
    for i in range(len(data)):
        label = tree.predict(data[i][:-1])
        if data[i][-1] == label:
            correct += 1
    print(total, correct, correct / total)
