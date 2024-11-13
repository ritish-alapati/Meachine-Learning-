from Q1a import *
from Q1d import remove_age
from Q2b import *

X, y = remove_age('train.txt', mode='train')

model = NaiveBayesClassifier(X, y)
