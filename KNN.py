import numpy as np
import pandas as pd

dataset = pd.read_csv('KNN.csv');

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,2].values

print(x)
print(y)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(x,y)

x_test = np.array([6,6])
ypred = classifier.predict([x_test])
print(ypred)

classifier = KNeighborsClassifier(n_neighbors = 3, weights = 'distance')
classifier.fit(x,y)

x_test = np.array([6,4])
ypred = classifier.predict([x_test])
print(ypred)
