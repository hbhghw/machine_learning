import math
eps = 1e-8

def readData(file='data.txt'):
    data = []
    with open(file) as f:
        for line in f.readlines():
            l = line.strip().split()[1:]
            data.append(l)
    return data


def calculateInfo(data, index):
    dict = {}
    for d in data:
        if d[index] not in dict.keys():
            dict[d[index]] = [0, 0]
        if d[-1] == 'yes':
            dict[d[index]][0] += 1
        else:
            dict[d[index]][1] += 1
    info = 0
    for v in dict.values():
        total = v[0] + v[1]
        info -= v[0] / total * math.log2(v[0] / total+eps) + v[1] / total * math.log2(v[1] / total+eps)
    #return info #-->ID3
    return info/len(dict) #-->C45

def splitData(data,index):
    dict = {}
    for d in data:
        if d[index] not in dict.keys():
            dict[d[index]] = [d]
        else:
            dict[d[index]].append(d)
    return dict

def getSplitIndex(data,remain_index):
    mininfo = math.inf
    index = -1
    for i in remain_index:
        info = calculateInfo(data, i)
        if info < mininfo:
            mininfo = info
            index = i
    return index

def isallSame(data):
    if len(data)==0:
        return True
    d0 = data[0][-1]
    for d in data[1:]:
        if d[-1]!=d0:
            return False
    return True

class Node:
    def __init__(self):
        self.splitIndex = -1
        self.arrivedValue = ""
        self.children = []
        self.label = ""

def generateTree(data,root,remain_index):
    if isallSame(data):
        root.label = data[0][-1]
        return root
    index = getSplitIndex(data,remain_index)
    if index == -1:
        return root
    root.splitIndex = index
    remain_index.remove(index)
    datadict = splitData(data,index)
    for k,d in datadict.items():
        node = Node()
        node.arrivedValue = k
        generateTree(d,node,remain_index)
        root.children.append(node)
    return root

def show(tree,attributes,depth):
    if tree.arrivedValue:
        print("\t" * depth, tree.arrivedValue)
        depth+=1
    if(tree.splitIndex!=-1):
        print("\t"*depth,attributes[tree.splitIndex])
        for node in tree.children:
            show(node,attributes,depth+1)
    if tree.label:
        print("\t"*depth,tree.label)

def predict(data,root):
    while(root.label==""):
        v = data[root.splitIndex]
        for child in root.children:
            if child.arrivedValue == v:
                root = child
                break
    return root.label

if __name__=='__main__':
    root = Node()
    data = readData()
    tree = generateTree(data[1:],root,list(range(len(data[0])-1)))

    show(tree,data[0],0)
    #note:data[0] is attribute line
    for d in data[1:]:
        print(predict(d,tree))

