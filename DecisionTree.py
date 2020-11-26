import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

df=pd.read_csv("tree.csv")
x=df.iloc[:,:-1]
y=df.iloc[:,5]


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

X=x.apply(le.fit_transform)
print(X)

xx=X.iloc[:,1:5]
dec=DecisionTreeClassifier()
clf=dec.fit(xx,y)

X_in=np.array([1,1,0,0])
print(clf.predict([X_in]))
dot_data = tree.export_graphviz(clf, out_file=None)
print(dot_data)
	

