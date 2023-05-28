# Car-Price-Prediction
#Helps in predicting or analyzing the car price based on different parameters 

import pandas as pd
import numpy as np
df = pd.read_csv('CarPrice.csv')
df
## DATA PREPROCESSING ##
df.head()
df.tail()
df.info()
df.shape
df.columns
df.duplicated().sum()
df.isnull().sum()
import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(df)
## CLASSIFICATION  OF CATEGORICAL DATA ##

print(df.fueltype.value_counts())
print(df.aspiration.value_counts())
print(df.doornumber.value_counts())
print(df.carbody.value_counts())
print(df.drivewheel.value_counts())
print(df.fuelsystem.value_counts())
print(df.cylindernumber.value_counts())
df['horsepower'].value_counts()
df['stroke'].value_counts()
df['compressionratio'].value_counts()
df['citympg'].value_counts()
df['highwaympg'].value_counts()
## CHANGING THE CATEGORICAL ATTRIBUTES INTO NUMERIC DATA FOR BETTER ANALYSIS ##

df.replace({'fueltype':{'gas':0,'diesel':1}},inplace=True)
df.replace({'aspiration':{'std':0,'turbo':1}},inplace=True)
df.replace({'doornumber':{'two':0,'four':1}},inplace=True)
df.replace({'carbody':{'convertible':0,'hatchback':1,'sedan':2, 'wagon':3}},inplace=True)
df.replace({'drivewheel':{'rwd':0,'fwd':1,'4wd':2}},inplace=True)
df.replace({'fuelsystem':{'mpfi':0,'2bbl':1,'1bbl':2,'mfi':3, 'spf1':4, 'idi':5}},inplace=True)
## hot encoding ##
df = pd.get_dummies(df,drop_first=True)
df
## Splitting the data ##

x = df.drop(['price'], axis=1) 
y = df['price']
print(len(x), len(y))
print(x)
print(y)
print(x.shape)
print(y.shape)
## Training and Test Data ##
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=1/10,random_state=0)
## Linear Regression ##

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
model.score(x_test,y_test)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
t_data_predic = regressor.predict(x_train)
## Error Calculation ##

from sklearn import metrics
error_score = metrics.r2_score(y_train, t_data_predic)
print("R squared Error : ", error_score)
## Plotting THE data ## 

plt.scatter(y_train, t_data_predic)
plt.xlabel("Actual Price of Car")
plt.ylabel("Predicted Price of Car")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()
 ## prediction on Training data ##
t_data_predic = regressor.predict(x_test)
# R squared Error ##
error_score = metrics.r2_score(y_test, t_data_predic)
print("R squared Error : ", error_score)
plt.scatter(y_test, t_data_predic)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()





