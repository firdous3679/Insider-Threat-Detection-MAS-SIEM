from __future__ import annotations
import math
from collections import Counter

class SparseMatrix:
    def __init__(self, rows, n_features):
        self.rows = rows
        self.n_features = n_features
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return SparseMatrix(self.rows[idx], self.n_features)
        return self.rows[idx]

class SimpleTfidfVectorizer:
    def __init__(self, max_features=5000, ngram_range=(1,2), min_df=2, stop_words=None, extra_phrases=None):
        self.max_features=max_features; self.ngram_range=ngram_range; self.min_df=min_df
        self.stop_words = set(['the','a','an','and','or','to','of','in','for','on','at','is','are','be','with','this','that','it','as','by','from']) if stop_words=='english' else set(stop_words or [])
        self.extra_phrases=list(extra_phrases or [])
        self.vocabulary_={}
        self.idf_={}
    def _tokenize(self, text):
        text_lower=text.lower()
        toks=[t for t in __import__('re').findall(r'\b\w+\b', text_lower) if t not in self.stop_words]
        feats=[]
        if self.ngram_range[0]<=1:
            feats.extend(toks)
        if self.ngram_range[1]>=2:
            feats.extend([toks[i]+' '+toks[i+1] for i in range(len(toks)-1)])
        for phrase in self.extra_phrases:
            if phrase in text_lower:
                feats.append(f"__phrase__:{phrase}")
        return feats
    def fit(self, texts):
        df=Counter()
        docs=[]
        for text in texts:
            feats=set(self._tokenize(text))
            docs.append(feats)
            df.update(feats)
        items=[(term,c) for term,c in df.items() if c>=self.min_df]
        items.sort(key=lambda x:(-x[1], x[0]))
        items=items[:self.max_features]
        n_docs=max(len(texts),1)
        self.vocabulary_={term:i for i,(term,_) in enumerate(items)}
        self.idf_={term:(math.log((1+n_docs)/(1+dfreq))+1.0) for term,dfreq in items}
        return self
    def transform(self, texts):
        rows=[]
        for text in texts:
            feats=self._tokenize(text)
            counts=Counter(f for f in feats if f in self.vocabulary_)
            for term in list(counts):
                if term.startswith("__phrase__:"):
                    counts[term] *= 4
            total=sum(counts.values()) or 1
            row={self.vocabulary_[term]:(cnt/total)*self.idf_[term] for term,cnt in counts.items()}
            rows.append(row)
        return SparseMatrix(rows, len(self.vocabulary_))
    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

class SimpleLinearClassifier:
    def __init__(self):
        self.log_prob={0:{},1:{}}
        self.log_prior={0:0.0,1:0.0}
    def fit(self, X, y):
        import math
        class_counts=Counter(y)
        feature_totals={0:Counter(),1:Counter()}
        total_weight={0:0.0,1:0.0}
        for row,label in zip(X.rows, y):
            for idx,val in row.items():
                feature_totals[int(label)][idx]+=val
                total_weight[int(label)]+=val
        n_features=X.n_features
        for c in [0,1]:
            self.log_prior[c]=math.log((class_counts[c]+1)/(len(y)+2))
            denom=total_weight[c]+n_features
            self.log_prob[c]={i:math.log((feature_totals[c][i]+1)/denom) for i in range(n_features)}
        return self
    def _score_row(self, row):
        import math
        scores={}
        for c in [0,1]:
            s=self.log_prior[c]
            for idx,val in row.items():
                s+=val*self.log_prob[c].get(idx, -20.0)
            scores[c]=s
        return scores
    def predict_proba(self, X):
        import math, numpy as np
        probs=[]
        rows = X.rows if hasattr(X,'rows') else [X]
        for row in rows:
            scores=self._score_row(row)
            m=max(scores.values())
            e0=math.exp(scores[0]-m); e1=math.exp(scores[1]-m)
            z=e0+e1
            probs.append([e0/z,e1/z])
        return np.array(probs)
    def predict(self, X):
        import numpy as np
        p=self.predict_proba(X)[:,1]
        return np.array([1 if v>=0.5 else 0 for v in p])
