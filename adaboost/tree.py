import math
import numpy as np


class Node:
    def __init__(self):
        self.index = None
        self.threshold = None
        self.leftChild = None
        self.rightChild = None
        self.isLeaf = False
        self.score = None


class Tree:  # decision trees as weak classifiers
    def __init__(self, max_depth=5):
        self.root = Node()
        self.max_depth = max_depth
        self.weight = 0
        self.updateSampleWeight = 0

    def isAllSame(self, samples):
        if len(samples) <= 1:
            return True

        return len(set([sample[-1] for sample in samples])) == 1

    def nodeScoreAndInfo(self, samples, weights):
        # return: node score and info at this node
        posW = sum([weights[i] for i in range(len(samples)) if samples[i][-1] == 1])
        negW = sum([weights[i] for i in range(len(samples)) if samples[i][-1] == -1])
        # node score is computed as (posW - negW) / (posW + negW),so if final score>0,we thought it's a positive sample
        # alternative ways are >>> posW - negW <<<  or
        # >>> 1 if posW>negW else -1 <<<,etc.
        return (posW - negW) / (posW + negW), -posW / (posW + negW) * math.log2(posW / (posW + negW) + 1e-9) - negW / (
                posW + negW) * math.log2(negW / (posW + negW) + 1e-9)

    def computeInfo(self, samples, weights, index):
        #also see computeInfo in stump.py
        ziped = zip(samples, weights)
        ziped = sorted(ziped, key=lambda item: item[0][index])
        samples, weights = [item[0] for item in ziped], [item[1] for item in ziped]
        threshold = None
        minInfo = 9999
        leftPos = 0
        leftNeg = 0
        rightPos = sum([weights[i] for i in range(len(samples)) if samples[i][-1] == 1])
        rightNeg = sum([weights[i] for i in range(len(samples)) if samples[i][-1] == -1])
        #total weight = rightpos+rightneg = 1
        for i in range(len(samples) - 1):
            if samples[i][-1] == 1:
                leftPos += weights[i]
                rightPos -= weights[i]
            else:
                leftNeg += weights[i]
                rightNeg -= weights[i]
            #this way is not good
            # leftInfo = -leftPos / (leftPos + leftNeg) * math.log2(leftPos / (leftNeg + leftPos) + 1e-9) \
            #            - leftNeg / (leftPos + leftNeg) * math.log2(leftNeg / (leftPos + leftNeg) + 1e-9)
            # rightInfo = -rightPos / (rightPos + rightNeg) * math.log2(rightPos / (rightPos + rightNeg) + 1e-9) \
            #            - rightNeg / (rightPos + rightNeg) * math.log2(rightNeg / (rightPos + rightNeg) + 1e-9)
            # if (leftPos + leftNeg) * leftInfo + (rightPos + rightNeg) * rightIno < minInfo:
            #     minInfo = leftInfo + rightInfo
            #     threshold = samples[i][index]
            leftfit = (leftPos-leftNeg)/(leftPos+leftNeg)
            rightfit = (rightPos-rightNeg)/(rightPos+rightNeg)
            leftinfo = leftPos*(leftfit-1)*(leftfit-1)+leftNeg*(leftfit+1)*(leftfit+1)
            rightinfo = rightPos*(rightfit-1)*(rightfit-1)+rightNeg*(rightfit+1)*(rightfit+1)
            if leftinfo+rightinfo<minInfo:
                minInfo = leftinfo + rightinfo
                threshold = samples[i][index]
        return minInfo, threshold

    def getBestSplitIndex(self, samples, weights, remain_index, nodeInfo):
        # get the best split index and relative threshold
        bestIndex = -1
        minInfo = nodeInfo
        threshold = None
        for index in remain_index:
            info, th = self.computeInfo(samples, weights, index)
            if info < minInfo:
                minInfo = info
                bestIndex = index
                threshold = th
        return bestIndex, threshold

    def splitSamples(self, samples, weights, index, threshold):
        # split samples at feature[index] by threshold
        left, right, leftW, rightW = [], [], [], []
        for i in range(len(samples)):
            if samples[i][index] <= threshold:
                left.append(samples[i])
                leftW.append(weights[i])
            else:
                right.append(samples[i])
                rightW.append(weights[i])
        return left, leftW, right, rightW

    def train(self, samples, weights, remain_index=None):
        # sample format:[feature1,f2,f3,...fn,label]
        if remain_index is None:
            remain_index = list(range(len(samples[0]) - 1))
        self._train(self.root, samples, weights, remain_index, 0)
        self.update(samples, weights)

    def _train(self, node, samples, weights, remain_index, depth):
        if self.isAllSame(samples):
            node.isLeaf = True
            if len(samples)==0:
                node.score = 0
            else:
                node.score = samples[0][-1]
            return
        score, nodeInfo = self.nodeScoreAndInfo(samples, weights)
        if len(remain_index) == 0 or depth == self.max_depth:
            node.isLeaf = True
            node.score = score
            return
        bestSplitIndex, threshold = self.getBestSplitIndex(samples, weights, remain_index, nodeInfo)
        if bestSplitIndex == -1:
            node.isLeaf = True
            node.score = score
            return

        node.index = bestSplitIndex
        node.threshold = threshold
        leftSamples, leftWeights, rightSamples, rightWeights = self.splitSamples(samples, weights, bestSplitIndex,
                                                                                 threshold)
        remain_index.remove(bestSplitIndex)
        leftNode = Node()
        rightNode = Node()
        self._train(leftNode, leftSamples, leftWeights, remain_index, depth + 1)
        self._train(rightNode, rightSamples, rightWeights, remain_index, depth + 1)
        node.leftChild = leftNode
        node.rightChild = rightNode

    def predict(self, sample):
        return self.weight * self._predict(sample)

    def _predict(self, sample):
        node = self.root
        while not node.isLeaf:
            if sample[node.index] < node.threshold:
                node = node.leftChild
            else:
                node = node.rightChild
        return node.score

    def update(self, samples, weights):
        acc = []
        err = []
        for i in range(len(samples)):
            score = self._predict(samples[i])
            if score * samples[i][-1] > 0:
                acc.append(i)
            else:
                err.append(i)
        accW = sum([weights[i] for i in acc])
        errW = sum([weights[i] for i in err])
        accuracy = accW / (accW + errW)
        print("Single tree accuracy:", accuracy)
        self.weight = 1 / 2 * math.log2(accuracy / (1 - accuracy))
        # update samples' weight
        for i in acc:
            weights[i] = weights[i] * pow(math.e, -self.weight)
        for i in err:
            weights[i] = weights[i] * pow(math.e, self.weight)
        sumw = sum(weights)
        weights = [weight / sumw for weight in weights]
        self.updateWeights = weights


if __name__ == '__main__':
    samples = np.load('data.npy')
    weights = [1 / samples.shape[0] for _ in range(samples.shape[0])]
    tree = Tree()
    tree.train(samples, weights)
    correct = 0
    for sample in samples:
        score = tree.predict(sample)
        if score * sample[-1] > 0:
            correct += 1
    print('acc:', correct / samples.shape[0])
