from sklearn._base import SimpleLinearClassifier
class RandomForestClassifier(SimpleLinearClassifier):
    def __init__(self, n_estimators=100, max_depth=None, random_state=None, n_jobs=None):
        super().__init__()
        self.n_estimators=n_estimators
        self.max_depth=max_depth
class IsolationForest:
    def __init__(self, *args, **kwargs):
        pass
