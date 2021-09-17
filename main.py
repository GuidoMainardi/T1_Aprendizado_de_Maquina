from KNearestNeighbor import KNearestNeighbor
import pandas as pd



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = train.values[:,:2]
y_train = train.values[:,2:]
X_test = test.values

knn = KNearestNeighbor(neighbors=7, policy='KDTree')

knn.fit(X_train, y_train)

# pred = knn.predict(X_test)

#print(pred)
#sns.jointplot(x='X', y='Y', data=df, hue='class', palette='rainbow')
#plt.show()