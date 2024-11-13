import numpy as np

def class_seperator(X, y):
    cls_sep = {}
    for idx in range(len(X)):
        vec = X[idx]
        cls_val = y[-1]
        if cls_val not in cls_sep:
            cls_sep[cls_val] = list()
        cls_sep[cls_val].append(vec)
    return cls_sep

# def get_mean(data):
#     return sum(data) / float(len(data))

# def get_std(data):
#     mean = get_mean(data)
#     var = sum([(x - mean)**2 for x in data]) / float(len(data)-1)
#     return np.sqrt(var)

def statistics(data):
    for fet in zip(*data):
        yield {
            'std': np.std(fet),
            'mean': np.mean(fet)
        }

def distrib(X, mean, std):
    expt = np.exp(-(X-mean)**2 / (2*std**2))
    return expt / (np.sqrt(2*np.pi)*std)

def class_summ(X, y):
    sep_cls = class_seperator(X, y)
    cls_summ = {}
    for lbl, fet in sep_cls.items():
        cls_summ[lbl] = {
            'prior': len(fet) / len(X),
            'summary': [summ for summ in statistics(fet)]
        }
    return cls_summ

def class_proba(X, smr):
    max_apriori = []
    for row in X:
        jnt_prob = {}
        for lbl, fets in smr.items():
            t_fet = len(fets['summary'])
            lkld = 1

            for idx in range(t_fet):
                ft = row[idx]
                mean = fets['summary'][idx]['mean']
                std = fets['summary'][idx]['std']
                norm_prob = distrib(X, mean, std)
                lkld *= norm_prob
            prior = fets['prior']
            jnt_prob[lbl] = prior * lkld
        mapi = max(jnt_prob, key=jnt_prob.get)
        max_apriori.append(mapi)
    return max_apriori

def accuracy_met(y_tr, y_pr):
    true_pos = 0
    for yt, yp in zip(y_tr, y_pr):
        if yp == yt:
            true_pos += 1
    return true_pos / len(y_tr)