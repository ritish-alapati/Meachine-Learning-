# TODO: Implement leave-one-out strategy for evaluation during training

# from Q1a import load_data, load_test
from Q1b import *
from Q1a import *
# import the KNN algorithm instead of creating a new one
from knn_algo import *

X_train, y_train =  load_data('train.txt')
tr, ts = load_data('train.txt')
tr_t, ts_t, tr_val, ts_val =tr[:110, :], tr[110:, :], ts[:110], ts[110:]
tst = load_test('test.txt')
k = 5
# t_dat = np.array(list(map(lambda x: float(x), input("Enter test data point(space separated): ").split(' '))))

preds = []
preds = knn_model(tr_t, tr_val, ts_t, k)
acc = accuracy(ts_val, preds)
print(acc)