import numpy as np

from Q1c import *
from Q1a import load_data, load_test
from knn_algo import *


# remove the age column
def remove_age(data, mode='train'):
    if mode:
        x, y = load_data(data)
        return np.delete(x, (2), axis=1), np.delete(y, (2))
    else:
        y = load_test(data)
        return np.delete(y, (2))

# X_train, y_train =  remove_age('train.txt')
tr, ts = remove_age('train.txt')
tr_t, ts_t, tr_val, ts_val =tr[:110, :], tr[110:, :], ts[:110], ts[110:]
tst = load_test('test.txt')
k = [1, 3, 5, 7, 9, 11]
# t_dat = np.array(list(map(lambda x: float(x), input("Enter test data point(space separated): ").split(' '))))

preds = []
for k in k:
    preds = knn_model(tr_t, tr_val, ts_t, k)
    print(preds)





