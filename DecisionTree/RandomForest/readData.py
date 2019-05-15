def getData(file='data.txt'):
    data = []
    labels = {}
    nlabel = 0
    with open(file) as f:
        for line in f.readlines():
            line = line.strip().split(',')
            _d = [float(i) for i in line[:-1]]
            if line[-1] not in labels.keys():
                labels[line[-1]] = nlabel
                nlabel += 1
            _d.append(labels[line[-1]])
            data.append(_d)
    return data,labels