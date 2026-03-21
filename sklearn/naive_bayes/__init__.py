from sklearn._base import SimpleLinearClassifier
class MultinomialNB(SimpleLinearClassifier):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha=alpha
