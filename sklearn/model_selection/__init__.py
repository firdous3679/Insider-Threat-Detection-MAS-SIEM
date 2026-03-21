from __future__ import annotations
import random
import numpy as np

def train_test_split(*arrays, test_size=0.25, random_state=None, stratify=None):
    n=len(arrays[0])
    rng=random.Random(random_state)
    indices=list(range(n))
    if stratify is None:
        rng.shuffle(indices)
        test_n=int(round(n*test_size))
        test_idx=indices[:test_n]
        train_idx=indices[test_n:]
    else:
        groups={}
        for i,label in enumerate(stratify):
            groups.setdefault(int(label), []).append(i)
        train_idx=[]; test_idx=[]
        for g in groups.values():
            rng.shuffle(g)
            test_n=max(1, int(round(len(g)*test_size)))
            test_idx.extend(g[:test_n]); train_idx.extend(g[test_n:])
        rng.shuffle(train_idx); rng.shuffle(test_idx)
    out=[]
    for arr in arrays:
        arr_list=list(arr)
        out.append([arr_list[i] for i in train_idx])
        out.append([arr_list[i] for i in test_idx])
    return tuple(out)

class StratifiedKFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits=n_splits; self.shuffle=shuffle; self.random_state=random_state
    def split(self, X, y):
        y=list(y)
        groups={}
        for i,label in enumerate(y):
            groups.setdefault(int(label), []).append(i)
        rng=random.Random(self.random_state)
        if self.shuffle:
            for g in groups.values(): rng.shuffle(g)
        folds=[[] for _ in range(self.n_splits)]
        for g in groups.values():
            for idx,i in enumerate(g): folds[idx % self.n_splits].append(i)
        for k in range(self.n_splits):
            val_idx=sorted(folds[k])
            train_idx=sorted(i for j,f in enumerate(folds) if j!=k for i in f)
            yield np.array(train_idx), np.array(val_idx)
