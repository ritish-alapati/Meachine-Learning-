import numpy as np
from knn_algo import *
from Q1a import *

def accuracy(ytest, preds):
    correct = ytest == preds
    accy = (np.sum(correct) / ytest.shape[0]) * 100
    return accy


if __name__=='__main__':
    X_train, y_train =  load_data('train.txt')
    tr, ts = load_data('train.txt')
    tr_t, ts_t, tr_val, ts_val =tr[:110, :], tr[110:, :], ts[:110], ts[110:]
    tst = load_test('test.txt')
    k = [1, 3, 5, 7, 9, 11]
    # t_dat = np.array(list(map(lambda x: float(x), input("Enter test data point(space separated): ").split(' '))))

    preds = []
    for k in k:
        preds = knn_model(tr_t, tr_val, ts_t, k)
        acc = accuracy(ts_val, preds)
        print(f"K: {k} : Accuracy: {acc}")
