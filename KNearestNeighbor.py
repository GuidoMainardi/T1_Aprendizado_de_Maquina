import numpy as np
from heapq import heappush, heappop


class KNearestNeighbor:

    # Inicialize KNN
    def __init__(self, neighbors=1, policy='BruFor'):
        self.neighbors = neighbors
        if policy == 'BruFor' or policy == 'KDTree':
            self.policy = policy
        else:
            print(f'invalid policy: {policy}, \'BruFor\' policy set by default')
            self.policy = 'BruFor'

     
    # Train the model
    def fit(self, X, y):
        self.values = X
        self.classes = y


    # predict the values
    def predict(self, X):
        if self.policy == 'BruFor':
            return self.BruFor_pred(X)
        else:
            return self.KDTree_pred(X)


    def mode(self, array):
        values, counts = np.unique(array, return_counts=True)
        mode = counts.argmax()
        return values[mode]

    
    # Brute force policy predict
    def BruFor_pred(self, X):
        predicted_classes = []
        for point in X:
            nearests = []
            for i in range(len(self.values)):
                dist = self.distance(point, self.values[i])
                self.insert_distance_queue(nearests, dist, i)
            neighbors_classes = [self.classes[i] for i in [x[1] for x in nearests]]
            predicted_classes.append(self.mode(neighbors_classes))
        return predicted_classes
    

    # KDTree policy predict
    def KDTree_pred(self, X):
        pass


    def insert_distance_queue(self, nearests, dist, neghbor):
        if len(nearests) < self.neighbors:
            heappush(nearests, (dist * -1, neghbor))
        else:
            if dist < abs(nearests[0][0]):
                heappop(nearests)
                heappush(nearests, (dist * -1, neghbor))
        

    def distance(self, a, b):
        point_a = np.array(a)
        point_b = np.array(b)
        return np.linalg.norm(point_a - point_b)