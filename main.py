from KNearestNeighbor import KNearestNeighbor
import pandas as pd



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = train.values[:,:2]
y_train = train.values[:,2:]
X_test = test.values

brute_force = KNearestNeighbor(neighbors=1)
KDTree = KNearestNeighbor(neighbors=1, policy='KDTree')

brute_force.fit(X_train, y_train)
KDTree.fit(X_train, y_train)

print(brute_force.predict(X_test))
print(KDTree.predict(X_test))

# pred = knn.predict(X_test)

#print(pred)
#sns.jointplot(x='X', y='Y', data=df, hue='class', palette='rainbow')
#plt.show()