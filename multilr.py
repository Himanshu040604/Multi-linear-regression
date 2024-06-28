#Multiple regression
#step-1 import all the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from sklearn.linear_model import LinearRegression
import seaborn as sns

#import the data file
data = pd.read_csv(r"C:\Users\KIIT\Downloads\Multilr.csv")
print(data.head())
print(data.info())

#data preprocessing making the data in a usable format for model
print(data.dropna())
data.shape

#splitting data
output_col="price"
#input data 
x=data.iloc[:,data.columns!=output_col]#get the data in and to sliceup the data iloce is used basically when we want all rows and columns except the output defined columns
print(x.head())
#output data
y=data.loc[:, output_col]

#splitting this data into training and testing data we knwo how to do it by lr model creation 
#we will be using sklearn packages as it has everything pre-defined

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=10)#this 0.30 means we are refining the 30% of the data for test site and rest 70% will go in training site
'''by manualy adding the random data it will add biasing in the data here we used random_state which we make the random training data for our model at the time of training it '''

print(data.shape)

print(x_train.shape)

print(x_test.shape)# as we split the data for training and testing the size of the col and rows reduced


print(y_train.shape)

print(y_test.shape)

#linear regression with multiple inout paramets
#training
lr = LinearRegression()
lr.fit(x_train,y_train)
lr.coef_
lr.intercept_

#prediction
pred_value=lr.predict(x_test)
# we import this for the cost function calculation
from sklearn.metrics import mean_squared_error
cost=mean_squared_error(y_test,pred_value)

plt.plot(x_test,y_test,"*",color="green")
plt.plot(x_test,pred_value,"+",color="red")
plt.title("performance testing")
plt.xlabel("input")
plt.ylabel("output")
plt.show
#residual assumption
residuals=y_test-pred_value
y_test
plt.scatter(pred_value,residuals)
plt.xlabel("y_pred value")
plt.ylabel("residuals value")
plt.show()

#normality assumption
sns.distplot(residuals)# to get a histogram and graph at the same time 

