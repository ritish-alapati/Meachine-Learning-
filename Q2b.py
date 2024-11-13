import numpy as np

from Q1a import *
from Q1c import *
from Q1d import *
from Q2a import *

class NaiveBayesClassifier:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def class_seperator(self):
        cls_sep = {}
        for idx in range(len(self.X)):
            vec = self.X[idx]
            cls_val = self.y[-1]
            if cls_val not in cls_sep:
                cls_sep[cls_val] = list()
            cls_sep[cls_val].append(vec)
        return cls_sep

    def info_stat(self):
        for fet in zip(*self.X):
            yield {
                'std': np.std(fet),
                'mean': np.mean(fet)
            }

    def fit(self):
        sep_cls = class_seperator(self.X, self.y)
        self.cls_summ = {}
        for lbl, fet in sep_cls.items():
            self.cls_summ[lbl] = {
                'prior': len(fet) / len(self.X),
                'summary': [summ for summ in statistics(fet)]
            }
        return self.cls_summ

    def distrib(self, mean, std):
        expt = np.exp(-(self.X-mean)**2 / (2*std**2))
        return expt / (np.sqrt(2*np.pi)*std)

    def predict_prob(self):
        max_apriori = []
        for row in self.X:
            jnt_prob = {}
            for lbl, fets in self.cls_summ.items():
                t_fet = len(fets['summary'])
                lkld = 1

                for idx in range(t_fet):
                    ft = row[idx]
                    mean = fets['summary'][idx]['mean']
                    std = fets['summary'][idx]['std']
                    norm_prob = distrib(self.X, mean, std)
                    lkld *= norm_prob
                prior = fets['prior']
                jnt_prob[lbl] = prior * lkld
            mapi = max(jnt_prob, key=jnt_prob.get)
            max_apriori.append(mapi)
        return max_apriori

    def accuracy(y_tr, y_pr):
        true_pos = 0
        for yt, yp in zip(y_tr, y_pr):
            if yp == yt:
                true_pos += 1
        return true_pos / len(y_tr)
