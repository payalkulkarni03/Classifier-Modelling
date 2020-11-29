import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv("1.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x,y)
print "Accuracy:" ,regressor.score(x,y)*100
y_pred=regressor.predict([[8]])
print y_pred
hours=int(input('Enter no of hours'))
eq=regressor.coef_*hours+regressor.intercept_
print 'y=%f*%f+%f' %(regressor.coef_,hours,regressor.intercept_)
print "Risk Score:",eq[0]
plt.plot(x,y,'o')
plt.plot(x,regressor.predict(x));
plt.show()
