#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Predictation

# ### Data Set Winequality-red.csv From Kaggle 
# 

# In[31]:


#Importing the Dependecies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[32]:


# Data Collection
Wine_data = pd.read_csv("C:/Users/HP/Desktop/winequality-red.csv")
Wine_data.shape


# In[33]:


Wine_data.head()


# In[34]:


# For check messing Value
Wine_data.isnull().sum()


# In[35]:


# Data Analysis & Visulization 
#Statistical measure of dataset
Wine_data.describe()


# In[36]:


# Number of Values of Each Quality
sns.catplot(x='quality',data=Wine_data,kind='count')


# In[37]:


# Volatile Vs Quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='volatile acidity',data=Wine_data)


# In[38]:


#Citric Acid Vs Quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='citric acid',data = Wine_data)


# In[39]:


# Correlation
correlation = Wine_data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size:8'}, cmap='Blues')


# In[40]:


# Data Processing 
#Seprate the data & Label
X=Wine_data.drop('quality',axis=1)
print(X)


# # Label Banirization 
# 

# In[41]:


Y = Wine_data['quality'].apply(lambda y_value:1 if y_value>=7 else 0)
print(Y)


# In[42]:


#Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 3)
print(Y.shape, Y_train.shape,Y_test.shape)


# In[43]:


# Model Training
RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,Y_train)


# In[44]:


#model Evaluation
#Accuracy On Dataset
X_test_predictation = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_predictation,Y_test)
print("Accuracy: ",test_data_accuracy)


# In[45]:


#Building Predictive System 
input_data =(7.9,0.6,0.06,1.6,0.069,15.0,59.0,0.9964,3.3,0.46,9.4)


# In[46]:


#Changing the input data to numpy array
input_data_as_numpy_array = np.array(input_data)


# In[47]:


#Reshape the data as we are predicting the label for only one instance 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


# In[49]:


prediction = model.predict(input_data_reshaped)
print(prediction)


# In[51]:


if (prediction[0]==1):
    print("Bad Quality Wine")
else:
    print("Good Quality Wine")

