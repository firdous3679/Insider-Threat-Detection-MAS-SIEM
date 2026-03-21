from sklearn._base import SimpleLinearClassifier
class LogisticRegression(SimpleLinearClassifier):
    def __init__(self, max_iter=1000, solver=None, random_state=None, n_jobs=None, C=1.0):
        super().__init__()
