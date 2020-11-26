import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = np.array([[0.1,0.6],[0.15,0.71],[0.08,0.9],[0.16,0.85],[0.2,0.3],[0.25,0.5],[0.24,0.1],[0.3,0.2]])

centroids = np.array([[x[0][0],x[7][0]],[x[0][1],x[7][1]]]);

print("Centroids are : ")
print(centroids)

from sklearn.cluster import KMeans

model = KMeans(n_clusters = 2, init = centroids)
model.fit(x)
labels = model.labels_
print("Labels for points : ",labels)

print("P6 belongs to : ",labels[5])
print("No of population around cluster 2 is : ", np.sum(labels))

new_centroids = model.cluster_centers_

print("Previous centroids : ",centroids)
print("New centroids : ",new_centroids)

plt.scatter(x[:,0],x[:,1], color='b')
plt.scatter(new_centroids[0][0],new_centroids[0][1],marker="*",color='r')
plt.scatter(new_centroids[1][0],new_centroids[1][1],marker="^",color='g')

plt.show()
