#!/usr/bin/env python
# coding: utf-8

# I've imported necessary Python libraries for data analysis and machine learning models.

# In[13]:


import pandas as pd 
import warnings
import itertools
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
from sklearn.model_selection import train_test_split


# I've converted the date into numbers so it doesn't show any error. The reason I've taken date-time into account is because the cash flow depends on the time interval. We have to make sure if it's long term forecasting or short term forecasting. In this case, it's a short term forecasting.

# In[14]:


df = pd.read_csv('./to_community/anonymized_train_data.csv')
Ch = df[:55]
Ge = df[55:110]
Ir = df[110:165]
Sw = df[165:220]
US = df[220:]
US.head()


# In[15]:


Sw['Date'].min(), Sw['Date'].max()


# Function for dropping the Null values (rows where NaN exists)

# In[16]:


def preparation(dataf, features, target):
    dataf = dataf.dropna()
    
    X = dataf[features].copy()
    y = dataf[target].copy()
    
    return X, y


# In[17]:


features = ['Date','AP Adj','Cost']
target = ['Cash Flow']
X1, y1 = preparation(Ch, features, target)
X2, y2 = preparation(Ge, features, target)
X3, y3 = preparation(Ir, features, target)
X4, y4 = preparation(Sw, features, target)
X5, y5 = preparation(US, features, target)
X2.shape, y2.shape


# In[18]:


def clean_data(X, y, rstate):
    return train_test_split(X, y, test_size=0.09, random_state=rstate)


# In[19]:


X1_train, X1_test, y1_train, y1_test = clean_data(X1, y1, 90000)
print(X1_train.shape, y1_train.shape)
print(X1_test.shape, y1_test.shape)
X2_train, X2_test, y2_train, y2_test = clean_data(X2, y2, 90000)
X3_train, X3_test, y3_train, y3_test = clean_data(X3, y3, 90000)
X4_train, X4_test, y4_train, y4_test = clean_data(X4, y4, 90000)
X5_train, X5_test, y5_train, y5_test = clean_data(X5, y5, 90000)


# In[20]:


def train_regressor(X_train, y_train):
    from sklearn.ensemble import AdaBoostRegressor
    
  #  _regressor = AdaBoostRegressor()
    _regressor = LinearRegression ()
  #  _regressor = DecisionTreeRegressor(max_depth=20)
    _regressor.fit(X_train, y_train)
    return _regressor


# In[25]:


X1_test


# I've trained 5 different models for 5 different countries. Because they don't follow the same pattern but each country has a distinct pattern.The Root Mean Square Error has been shown additionally.

# In[21]:


model1 = train_regressor(X1_train, y1_train['Cash Flow'])
y1_prediction = model1.predict(X1_test)
rmse1 = sqrt(mean_squared_error(y_true = y1_test, y_pred = y1_prediction))
print(rmse1)
#except AssertionError as e: print("Keep trying - can you get an RMSE < %f" % threshold)
model2 = train_regressor(X2_train, y2_train['Cash Flow'])
y2_prediction = model2.predict(X2_test)
rmse2 = sqrt(mean_squared_error(y_true = y2_test, y_pred = y2_prediction))
print(rmse2)
model3 = train_regressor(X3_train, y3_train['Cash Flow'])
y3_prediction = model3.predict(X3_test)
rmse3 = sqrt(mean_squared_error(y_true = y3_test, y_pred = y3_prediction))
print(rmse3)
model4 = train_regressor(X4_train, y4_train['Cash Flow'])
y4_prediction = model4.predict(X4_test)
rmse4 = sqrt(mean_squared_error(y_true = y4_test, y_pred = y4_prediction))
print(rmse4)
model5 = train_regressor(X5_train, y5_train['Cash Flow'])
y5_prediction = model5.predict(X5_test)
rmse5 = sqrt(mean_squared_error(y_true = y5_test, y_pred = y5_prediction))
print(rmse5)


# ### The following values have been taken from China.xlsx, Germany.xlsx, Ireland.xlsx, Sw.xlsx, US.xlsx files
# The AP Adj, Cost data of each file have been calculated by averaging the values of past 3 available days. The cash flow is to be predicted by our ML model.

# In[39]:


d1={'Date':[43678.00,43709.0,43739.00,43770.00,43800.00],'AP Adj':[0.873318,0.841404,0.734025,0.816249,0.797226],'Cost':[0.007418,0.1058,-0.02507,0.029381,0.036702]}
d1=pd.DataFrame(data=d1)
d2={'Date':[43678.00,43709.0,43739.00,43770.00,43800.00],'AP Adj':[0.906075777,0.892320106,0.819211283,0.872535722,0.861355704],'Cost':[0.148744502,0.140467015,0.413289382,0.234166966,0.262641121]}
d2=pd.DataFrame(data=d2)
d3={'Date':[43678.00,43709.0,43739.00,43770.00,43800.00],'AP Adj':[1.199977949,1.186726794,1.219807063,1.202170602,1.202901486],'Cost':[0.064738533,0.067821651,0.321416976,0.15132572,0.180188116]}
d3=pd.DataFrame(data=d3)
d4={'Date':[43678.00,43709.0,43739.00,43770.00,43800.00],'AP Adj':[0.790942345,0.767446076,0.712290784,0.756893068,0.74554331],'Cost':[-0.213781256,-0.222236474,-0.234410606,-0.223476112,-0.22670773]}
d4=pd.DataFrame(data=d4)
d5={'Date':[43678.00,43709.0,43739.00,43770.00,43800.00],'AP Adj':[-0.445984661,-0.456103654,-0.447299181,-0.449795832,-0.451066222],'Cost':[-0.668328998,-0.579576009,-0.613963628,-0.620622878,-0.604720838]}
d5=pd.DataFrame(data=d5)


# The cash flow forecasting of 5 different countries from 8/1/2019 to 12/1/2019

# In[42]:


print('China',model1.predict(d1))
print('Germany',model2.predict(d2))
print('Ireland',model3.predict(d3))
print('Switzerland',model4.predict(d4))
print('USA',model5.predict(d5))


# In[ ]:




