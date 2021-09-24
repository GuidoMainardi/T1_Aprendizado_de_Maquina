import numpy as np
from heapq import heappush, heappushpop

class KNearestNeighbor:

    class Node:

        def __init__(self, point, dim):
            self.left = None
            self.right = None
            self.point = point
            self.dim = dim

    # Inicialize KNN
    def __init__(self, neighbors=1, policy='BruFor'):
        self.neighbors = neighbors
        if policy == 'KDTree':
            if neighbors != 1:
                print(f'KDTree policy only suports 1 nearest neighbor!')
                self.neighbors = 1 
            self.policy = policy
        elif policy == 'BruFor':
            self.policy = policy
        else:
            print(f'invalid policy: {policy}, \'BruFor\' policy set by default')
            self.policy = 'BruFor'

     
    # Train the model
    def fit(self, X, y):
        if self.policy == 'BruFor':
            self.values = np.array(X)
            self.classes = np.array(y)
            #print('Brute Force model Fitted!')
        else:
            array = np.c_[np.array(X), np.array(y)]
            self.root = self.build_tree(array, 0)
            #self.print_tree(self.root)
            #print('KDTree builded!')

            

    def build_tree(self, array, dim):
        # sort array by dim
        array = array[array[:, dim].argsort()]

        #split array in 2
        splited_array = np.array_split(array, 2)

        # Get this node value
        this = splited_array[0][-1]
        splited_array[0] = splited_array[0][:-1]
        node = self.Node(this, dim)

        # build next steps of tree
        if len(splited_array[0]):
            node.left = self.build_tree(splited_array[0], (dim + 1) % (len(this) - 1))
        if len(splited_array[1]):
            node.right = self.build_tree(splited_array[1], (dim + 1) % (len(this) - 1))

        return node
        
    
    def print_tree(self, node, level=0):
        if node:
            self.print_tree(node.left, level+1)
            print(" " * 20 * level + '->', f'{node.point}')
            self.print_tree(node.right, level+1)
    

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
            neighbors_classes = [self.classes[j] for j in [x[1] for x in nearests]]
            predicted_classes.append(self.mode(neighbors_classes))
        return predicted_classes
    
    def closer(self, point, p1, p2):
        if p1 is None:
            return p2
        elif p2 is None:
            return p1

        return p1 if self.distance(p1[:-1], point) < self.distance(p2[:-1], point) else p2
    
    def KDTree_pred(self, X):
        predicted_classes = []
        for point in X:
            predicted_classes.append(self.KDTree(self.root, point)[-1])
        return predicted_classes
    
    # KDTree policy predict
    def KDTree(self, root, point):
        if not root:
            return None
        
        if point[root.dim] < root.point[root.dim]:
            nearest = self.closer(point, self.KDTree(root.left, point), root.point)
            

            if self.distance(point, nearest[:-1]) > abs(point[root.dim] - root.point[root.dim]):
                nearest = self.closer(point, self.KDTree(root.right, point), nearest)

        else:
            nearest = self.closer(point, self.KDTree(root.right, point), root.point)

            if self.distance(point, nearest[:-1]) > abs(point[root.dim] - root.point[root.dim]):
                nearest = self.closer(point, self.KDTree(root.left, point), nearest)

        return nearest


    def insert_distance_queue(self, nearests, dist, neghbor):
        if len(nearests) < self.neighbors:
            heappush(nearests, (dist * -1, neghbor))
        else:
            if dist < abs(nearests[0][0]):
                heappushpop(nearests, (dist * -1, neghbor))
        

    def distance(self, a, b):
        point_a = np.array(a)
        point_b = np.array(b)
        return np.linalg.norm(point_a - point_b)