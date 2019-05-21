import numpy as np

labels = {
    'R':1,
    'M':-1
}
def getData(file='data.txt'):
    data = []
    with open(file) as f:
        line = f.readline()
        while line:
            line = line.strip().split(',')
            line[-1] = labels[line[-1]]
            line = [float(i) for i in line]
            data.append(line)
            line = f.readline()

    return np.asarray(data)

if __name__=='__main__':
    data = getData()
    np.save('data.npy',data)