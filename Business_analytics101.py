#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import matplotlib as plt
import numpy as np
import seaborn as sns


# In[18]:


df=pd.read_csv(r'C:\Users\Yimish Gedam\Desktop\MBA Sem 3\Business Analytics\Advertising.csv',index_col=0)


# In[19]:


df


# In[20]:


X=df.drop('Sales',axis=1)
y=df['Sales']


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[23]:


X_train.shape


# In[24]:


X_test.shape


# In[25]:


from sklearn.linear_model import LinearRegression


# In[26]:


LR=LinearRegression()


# In[27]:


lr=LR.fit(X_train,y_train)


# In[28]:


lr.intercept_


# In[29]:


lr.coef_

86% correct prediction
# In[31]:


lr.score(X_test,y_test)


# In[32]:


lr.score(X_train,y_train)


# In[33]:


lr.predict([[5,4,2]])


# In[34]:


from sklearn.model_selection import cross_val_score


# In[36]:


cross_val_score(lr,X,y,cv=5).mean()


# In[37]:


import joblib


# In[40]:


loaded_lr_model=joblib.dump(lr,'linear_regression_model.sav')


# In[41]:


loaded_lr_model=joblib.load('linear_regression_model.sav')


# In[42]:


loaded_lr_model.predict([[5,4,2]])


# In[ ]:





# In[ ]:




