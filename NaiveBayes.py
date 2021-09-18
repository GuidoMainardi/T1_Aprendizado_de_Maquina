import numpy as np

class NaiveBayes:
    

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    
    def fit(self, X, y):
        unique, counts = np.unique(np.array(y), return_counts=True)
        class_chances = dict(zip(unique, counts))