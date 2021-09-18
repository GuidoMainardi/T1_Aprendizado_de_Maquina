from NaiveBayes import NaiveBayes
import pandas as pd



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = train.values[:,:2]
y_train = train.values[:,2:]
X_test = test.values

nb = NaiveBayes()
nb.fit(X_train, y_train)
pred = nb.predict(X_test)
print(pred)