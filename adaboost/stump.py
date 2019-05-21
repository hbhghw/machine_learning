import math
import numpy as np


class Stump:  # single layer decision stump as weak classifiers
    def __init__(self):
        self.index = None
        self.threshold = None
        self.sign = None
        self.weight = 0
        self.updateWeights = 0

    def isAllSame(self, samples):
        if len(samples) <= 1:
            return True

        return len(set([sample[-1] for sample in samples])) == 1

    def computeInfo(self, samples, weights, index):
        '''
        also see computeInfo in tree.py
        '''
        ziped = zip(samples, weights)
        ziped = sorted(ziped, key=lambda item: item[0][index])
        samples, weights = [item[0] for item in ziped], [item[1] for item in ziped]
        threshold = None
        sign = None
        minerr = 9999
        leftPos = 0
        leftNeg = 0
        rightPos = sum([weights[i] for i in range(len(samples)) if samples[i][-1] == 1])
        rightNeg = sum([weights[i] for i in range(len(samples)) if samples[i][-1] == -1])
        #total weight = rightpos+rightneg = 1
        for i in range(len(samples)):
            if samples[i][-1] == 1:
                leftPos += weights[i]
                rightPos -= weights[i]
            else:
                leftNeg += weights[i]
                rightNeg -= weights[i]
            # first let left be label +1(positive),right be -1,so neg sample in left and pos sample in right are error classified,the err is
            err1 = leftNeg + rightPos
            # then let left be negative and right pos,the err is
            err2 = leftPos + rightNeg
            # choose the min error
            err = min(err1, err2)
            if err<minerr:
                minerr = err
                threshold = samples[i][index]
                sign = 'l' if err1<err2 else 'r' #left positive or right positive
        return minerr,threshold,sign

    def getBestSplitIndex(self, samples, weights, remain_index):
        # get the best split index and relative threshold
        bestIndex = -1
        minInfo = 9999
        threshold = None
        sign = None
        for index in remain_index:
            info, th,sig = self.computeInfo(samples, weights, index)
            if info < minInfo:
                minInfo = info
                bestIndex = index
                threshold = th
                sign = sig
        return bestIndex, threshold,sign

    def train(self, samples, weights, remain_index=None):
        # sample format:[feature1,f2,f3,...fn,label]
        if remain_index is None:
            remain_index = list(range(len(samples[0]) - 1))
        if self.isAllSame(samples):
            return
        bestSplitIndex, threshold,sign = self.getBestSplitIndex(samples, weights, remain_index)

        self.index = bestSplitIndex
        self.threshold = threshold
        self.sign = sign

        self.update(samples,weights)

    def predict(self, sample):
        return self.weight * self._predict(sample)

    def _predict(self, sample):
        if sample[self.index]<self.threshold:
            if self.sign == 'l':
                return 1
            else:
                return -1
        else:
            if self.sign == 'l':
                return -1
            else:
                return 1

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
        print("Single stump accuracy:", accuracy)
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
    stump = Stump()
    stump.train(samples, weights)
    correct = 0
    for sample in samples:
        score = stump.predict(sample)
        if score * sample[-1] > 0:
            correct += 1
    print('test acc:', correct / samples.shape[0])
