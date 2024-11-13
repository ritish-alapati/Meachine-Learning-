import numpy as np
from knn_algo import *

# loading the data provided from a .txt file to numpy arrays
# split the data into X and labels(y)
def load_data(data):
    lines, tars = [], []
    with open(data, 'r') as file:
        for line in file.readlines():
            line = line.translate({ord('('): None, ord(')'): None, ord(' '): None}).strip('\n').strip(' ')
            lbl = line[-1]
            lines.append([float(x) for x in line.split(',')[:-1]])
            tars.append(lbl)
    return np.array(lines), encode_class(tars)

# load the test data to numpy arrays
def load_test(data):
    lines = []
    with open(data, 'r') as file:
        for line in file.readlines():
            line = line.translate({ord('('): None, ord(')'): None, ord(' '): None}).strip('\n').strip(' ')
            lines.append([float(x) for x in line.split(',')])
    return np.array(lines)

# encode the labels from categorical to numeric
def encode_class(lbls):
    cls_map = dict(map(lambda v: (v, list(set(lbls)).index(v)), lbls))
    result = np.zeros(len(lbls))
    for idx, val in enumerate(lbls):
        ht_idx = cls_map[val]
        result[idx] = ht_idx
    return np.array(result)

# these distances functions are vectorized for speed
# functiomn to calculate the euclidean distances
def euclidean_distance(x, y):
    return -2 * x @ y.T + np.sum(y**2,axis=1) + np.sum(x**2,axis=1)[:, np.newaxis]

# function to calculate the manhattan distances
def manhattan_distance(x, y):
    return np.abs(-2 * x @ y.T + np.sum(y**2,axis=1) + np.sum(x**2,axis=1)[:, np.newaxis])

# function to calculate the minkowski distances
def minkowski_distance(x, y, order=3):
    return sum((np.abs(v1 - v2)**order for v1, v2 in zip(x, y)))**(1/order)



if __name__=='__main__':
    X, y = load_data('data1.txt')
    test = load_test('test.txt')
    k , metrics = [1, 3, 7], ['euclidean', 'manhattan']
    for k in k:
        for mt in metrics:
            ys = knn_model(X, y, test, k, mt)
            print(f"K: {k}, Metric: {mt}, Pred: {ys}")


    # print(f"{X} \n {y}")
