#!/usr/bin/env python
# coding: utf-8

# In[1]:


from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
# from sklearn.select_model import cross_validation
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import normalize
from sklearn import metrics
import seaborn as sns


# In[2]:


data = pd.read_csv("kc_house_data.csv")
data.head()


# In[3]:


df1=data[['price', 'bedrooms', 'bathrooms', 'sqft_living',
    'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
    'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
    'lat', 'long', 'sqft_living15', 'sqft_lot15']]
h = df1.hist(bins=25,figsize=(16,16),xlabelsize='10',ylabelsize='10',xrot=-15)
sns.despine(left=True, bottom=True)
[x.title.set_size(12) for x in h.ravel()];
[x.yaxis.tick_left() for x in h.ravel()];


# In[4]:


plt.figure(figsize=(8, 8))
plt.scatter(data['sqft_living'],data['price'], c='y')
plt.show()


# In[5]:


import seaborn as sns
fig, ax = plt.subplots(figsize=(16, 16))
sns.heatmap(data.corr(), annot=True)


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[7]:


X = data.drop(['price','id','date'], axis=1)
Y = data['price']


# In[8]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=3)
x_train


# In[9]:


def loadData(filename):
	data = np.genfromtxt(filename, delimiter=',', dtype=float, skip_header=1)
	return data
data = loadData("kc_house_data.csv")
data.shape


# In[10]:


impute = np.array([0] * len(data[0]))
impute[3] = 2  # num_bed
impute[4] = 2  # num_bath
impute[5] = 1  # living_space
impute[6] = 1  # living_lot
impute[7] = 1  # living_floors
impute[14] = 2  # year_built


# In[11]:


def mean_imputation_pure(pX_train, feature_to_impute):
    X_train = np.copy(pX_train)
    for i in range(0, len(feature_to_impute)):  # 0--> 21
        if feature_to_impute[i] == 0:
            continue
        non_zeros = 0
        for j in range(0, X_train.shape[0]):  # 0-->21613
            if X_train[j, i] != 0:
                non_zeros += 1
        mean = np.sum(X_train[:, i]) / float(non_zeros)
        for j in range(0, X_train.shape[0]):
            if X_train[j, i] == 0:
                X_train[j, i] = mean
    return X_train


# In[12]:


data_imputation=mean_imputation_pure(data,impute)


# In[13]:


xim=data_imputation[:,[3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20]]    # Biến độc lập
yim=data_imputation[:,2]     # Biến phụ thuộc


# In[14]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xim, yim, test_size = 0.2, random_state=42)


# In[15]:


import statsmodels.api as sm
x_train_with_intercept = sm.add_constant(x_train)
stats_model = sm.OLS(y_train, x_train_with_intercept)

fit_model = stats_model.fit()

print('fit_model', fit_model.summary())


# In[16]:


xim=data_imputation[:,[3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20]]    # Biến độc lập
yim=data_imputation[:,2] 


# In[17]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xim, yim, test_size = 0.2, random_state=42)


# In[18]:


import statsmodels.api as sm
x_train_with_intercept = sm.add_constant(x_train)
stats_model = sm.OLS(y_train, x_train_with_intercept)

fit_model = stats_model.fit()

print(fit_model.summary())


# In[19]:


reg_im = LinearRegression()
reg_im.fit(x_train, y_train)


# In[20]:


yim_predict = reg_im.predict(x_test)


# In[21]:


from sklearn.metrics import mean_squared_error
from math import sqrt


# In[22]:


print('My MAE: ' + str(metrics.mean_absolute_error(y_test, yim_predict)))

print('Sk MSE: ' + str(metrics.mean_squared_error(y_test, yim_predict)))

print('My RMSE: ' + str(np.sqrt(metrics.mean_absolute_error(y_test, yim_predict))))


def MAPE(predict, target):
    return (abs((target - predict) / target).mean()) * 100


print('My MAPE: ' + str(MAPE(yim_predict, y_test)))


# In[23]:


plt.figure(figsize = (30, 15))

plt.plot(yim_predict, label='Predicted')
plt.plot(y_test, label='Actual')

plt.ylabel("Price ($)")
plt.legend()
plt.show()


# In[24]:


data5 = pd.read_csv("kc_house_data_new.csv")
data5.head()


# In[25]:


df2=data5[['price', 'bedrooms', 'bathrooms', 'sqft_living',
    'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
    'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
    'lat', 'long', 'sqft_living15', 'sqft_lot15']]
h = df1.hist(bins=25,figsize=(16,16),xlabelsize='10',ylabelsize='10',xrot=-15)
sns.despine(left=True, bottom=True)
[x.title.set_size(12) for x in h.ravel()];
[x.yaxis.tick_left() for x in h.ravel()];

