import numpy as np
from math import pi, exp, sqrt, log
class NaiveBayes:
    

    def __init__(self, alpha=0.00001):
        self.alpha = alpha

    
    def fit(self, X, y):
        self.y = np.array(y)

        # calculate inicial classes probability
        unique, counts = np.unique(np.array(y), return_counts=True)
        self.classes = list(zip(unique, counts / len(y)))
        self.class_prob = list(map(lambda x: x[1], self.classes))
        #print(self.class_prob)

        # split dataset by class
        X = np.c_[np.array(X), np.array(y)]
        self.data_by_class = np.array([X[X[:,-1]==k] for k in np.unique(X[:,-1])], dtype=object)
        #print(self.data_by_class)

        # calculate the mean and the standard deviation for each feature in each class
        self.mean_std= [list(zip(np.mean(array[:,:-1], axis=0), np.std(array[:,:-1], axis=0))) for array in self.data_by_class]
        #[print(x) for x in self.mean_std]
    
    def likelihood(self, mean, std, value):
        if std:
            first_part = 1 / (sqrt(2*pi*(std**2)))
        else:
            return self.alpha
        second_part = exp((-((value-mean)**2))/(2*(std**2)))
        return first_part * second_part  + self.alpha

    def pred_class(self, point):
        #calculate prob of each class
        probs = []
        for feature in range(len(self.mean_std)):
            prob = log(self.classes[feature][1])
            for index in range(len(point)):
                prob += log(self.likelihood(self.mean_std[feature][index][0], self.mean_std[feature][index][1], point[index]))
            probs.append(prob)

        #return class with higher prob
        return self.classes[probs.index(max(probs))][0]



    def predict(self, X):
        predicted_class = []
        for point in X:
            predicted_class.append(self.pred_class(point))
        return predicted_class
