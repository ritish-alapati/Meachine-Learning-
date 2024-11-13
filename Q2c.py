from Q1c import *
from Q2b import *

X, y = load_data('train.txt')
test = load_test('test.txt')

model = NaiveBayesClassifier(X, y)

if __name__ == '__main__':
    model.fit()
    print(model.predict_prob())