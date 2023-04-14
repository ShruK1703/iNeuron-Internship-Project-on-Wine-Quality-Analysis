#!/usr/bin/env python
# coding: utf-8

# # WINE QUALTIY ANALYSIS- iNeuron Internship Project

# ## Presented by SHRUTI KHANDELWAL

# ### Dataset Information

# Two datasets are included, related to red and white vinho verde wine samples, from the north of Portugal. The goal is to model wine quality based on physicochemical tests (see [Cortez et al., 2009],
# 
# Data Set Information:
# 
# The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. For more details, consult: [Web Link] or the reference [Cortez et al., 2009]. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
# 
# These datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent or poor wines. Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods.
# 
# 
# Datasets downloaded from : https://archive.ics.uci.edu/ml/datasets/wine+quality

# ### Attribute Information:

# 
# 
# Input variables (based on physicochemical tests):
# 
# 1 - fixed acidity
# 
# 2 - volatile acidity
# 
# 3 - citric acid
# 
# 4 - residual sugar
# 
# 5 - chlorides
# 
# 6 - free sulfur dioxide
# 
# 7 - total sulfur dioxide
# 
# 8 - density
# 
# 9 - pH
# 
# 10 - sulphates
# 
# 11 - alcohol
# 
# Output variable (based on sensory data):
# 
# 12 - quality (score between 0 and 10)

# ### Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ### Data Preprocessing

# In[2]:


df = pd.read_csv(r'C:\Users\USER\Downloads\New folder\winequalityN.csv')
df.head()


# ##### Total number of rows and columns

# In[3]:


df.shape


# ##### Complete information about dataset

# In[4]:


df.info()


# In[5]:


df.columns


# In[6]:


df.dtypes


# In[7]:


#renaming columns
df.rename(columns = {'fixed acidity':'fixed_acidity', 'volatile acidity':'volatile_acidity', 'citric acid': 'citric_acid', 'residual sugar':'residual_sugar', 'free sulfur dioxide' : 'free_sulfur_dioxide', 'total sulfur dioxide': 'total_sulfur_dioxide'}, inplace = True)
df.head()


# ##### Checking for duplicates

# In[8]:


df.duplicated().sum()


# ##### Removing duplicates

# In[9]:


df = df.drop_duplicates()


# In[10]:


df.duplicated().sum()


# In[11]:


#After dropping duplicate values, our dataset has rows and columns : 
df.shape


# In[12]:


df


# ##### Checking for null values

# In[13]:


df.isnull().sum()


# ##### Filling out null values

# In[14]:


#Filling null values with mean/median
df['fixed_acidity'] = df['fixed_acidity'].fillna(df['fixed_acidity'].median())
df['fixed_acidity'].isnull().sum()


# In[15]:


df['volatile_acidity'].fillna(df['volatile_acidity'].mean(), inplace=True)
df['citric_acid'].fillna(df['citric_acid'].mean(), inplace=True)
df['residual_sugar'].fillna(df['residual_sugar'].mean(), inplace=True)
df['chlorides'].fillna(df['chlorides'].median(), inplace=True)
df['pH'].fillna(df['pH'].mean(), inplace=True)
df['sulphates'].fillna(df['sulphates'].median(), inplace=True)
df.isnull().sum()


# ##### Statistical information analysis

# In[16]:


df.describe()


# Some outliers can be seen in the statistical analysis, let's plot them and seen in details.
# 

# In[17]:


sns.set(rc = {'figure.figsize' : (20,10)})
sns.boxplot(data=df)
plt.show()


# Outliers can be seen in 3 columns: residual sugar, free sulfur dioxide, total sulfur dioxide. Let's remove them.

# In[18]:


#Removing outliers in free sulfur dioxide
lower = df['free_sulfur_dioxide'].mean()-3*df['free_sulfur_dioxide'].std()
upper = df['free_sulfur_dioxide'].mean()+3*df['free_sulfur_dioxide'].std()
df = df[(df['free_sulfur_dioxide']>lower) & (df['free_sulfur_dioxide']<upper)]

#Removing outliers in total sulfur dioxide
lower = df['total_sulfur_dioxide'].mean()-3*df['total_sulfur_dioxide'].std()
upper = df['total_sulfur_dioxide'].mean()+3*df['total_sulfur_dioxide'].std()
df = df[(df['total_sulfur_dioxide']>lower) & (df['total_sulfur_dioxide']<upper)]

#Removing outliers in residual sugar
lower = df['residual_sugar'].mean()-3*df['residual_sugar'].std()
upper = df['residual_sugar'].mean()+3*df['residual_sugar'].std()
df = df[(df['residual_sugar']>lower) & (df['residual_sugar']<upper)]


# In[19]:


df.describe()


# ### One hot encoding

# In[20]:


dum = pd.get_dummies(df.type, drop_first = True)
df = pd.concat([df, dum], axis = 1)
df.sample(5)


# In[21]:


df.drop('type', axis =1 , inplace = True)


# In[22]:


df.sample(10)


# Now 0 represents red wine, 1 represents white wine.

# ### EDA

# ##### Univariate Analysis

# In[23]:


print("Min quality unit is ", df.quality.min())
print("Max quality unit is ", df.quality.max())
print(df.quality.value_counts())


# In[24]:


df['residual_sugar'] = np.log(1 + df['residual_sugar'])


# In[25]:


sns.distplot(df['residual_sugar'])


# ##### Bivariate Analysis

# Relationship between Fixed Acidity and Quality.

# In[26]:


sns.barplot(x = df['quality'], y=df['fixed_acidity'])


# Relationship between Volatile Acidity and Quality.

# In[27]:


sns.barplot(x = df['quality'], y=df['volatile_acidity'])


# Relationship between Alcohol and Quality.

# In[28]:


sns.barplot(x = df['quality'], y=df['alcohol'])


# Relationship between Citric Acid and Quality.

# In[29]:


sns.barplot(x = df['quality'], y=df['citric_acid'])


# Relationship between Chlorides and Quality.

# In[30]:


sns.barplot(x = df['quality'], y=df['chlorides'])


# Relationship between Total Sulfur Dioxide and Quality.

# In[31]:


sns.barplot(x = df['quality'], y=df['total_sulfur_dioxide'])


# Relationship between Free Sulfur Dioxide and Quality.

# In[32]:


sns.barplot(x = df['quality'], y=df['free_sulfur_dioxide'])


# In[33]:


#finding relationship between features
corr= df.corr()
sns.set(font_scale = 1.5)
plt.figure(figsize=(20,10))
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,annot=True)
corr


# #### Mapping values for target variable

# In[34]:


df['quality']=df['quality'].map({3:'low', 4:'low', 5:'medium', 6:'medium', 7:'medium', 8:'high', 9:'high'})
df['quality']=df['quality'].map({'low':0,'medium':1,'high':2})


# ## Model Development

# In[35]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[36]:


X = df.drop(['quality','white'], axis=1)
y = df['quality'].apply(lambda y_value: 1 if y_value>= 1 else 0)


# In[37]:


print(X.head())
print(X.shape)


# In[38]:


print(y.sample(5))


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state =2)


# In[40]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# ### Logistic Regression

# In[41]:


from sklearn.linear_model import LogisticRegression


# In[42]:


lr = LogisticRegression()


# In[43]:


lr.fit(X_train, y_train)


# In[44]:


pr = lr.predict(X_test)
sc = accuracy_score(y_test, pr)
print("Accuracy is ", sc * 100)


# ### DecisionTreeClassifier

# In[45]:


from sklearn.tree import DecisionTreeClassifier


# In[46]:


mod = DecisionTreeClassifier()
print(mod.get_params())


# In[47]:


mod.fit(X_train, y_train)
y_pred_mod = mod.predict(X_test)
score_mod = accuracy_score(y_test, y_pred_mod)
print("Accuracy is ", score_mod * 100)


# ### RandomForestClassifier

# In[48]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
print(model.get_params())


# In[49]:


model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("Accuracy is ", score * 100)


# In[50]:


from sklearn.metrics import classification_report


# In[51]:


print(classification_report(y_test, y_pred))


# Choosing Random Forest Classifier for predicting quality of wine.

# ### Predicitng Wine Quality

# In[52]:


data = (9.1,0.25,0.39,2.1,0.036,30.0,124.0,0.99080,3.28,0.43,12.2)
array = np.asarray(data)
final = array.reshape(1,-1)
prediction = model.predict(final)
if (prediction[0] == 1):
    print('Wine Quality is Good')
else: 
    print('Wine Quality is Bad')
    
print(prediction)


# In[53]:


data = (4.2,0.215,0.23,1.808289,0.041,64.0,157.0,0.99688,3.42,0.44,8.0)
array = np.asarray(data)
final = array.reshape(1,-1)
prediction = model.predict(final)
if (prediction[0] == 1):
    print('Wine Quality is Good')
else: 
    print('Wine Quality is Bad')
    
print(prediction)


# In[54]:


data = (6.2,0.600,0.08,2.0,0.090,32.0,44.0,0.99490,3.45,0.58,10.5)
array = np.asarray(data)
final = array.reshape(1,-1)
prediction = model.predict(final)
if (prediction[0] == 1):
    print('Wine Quality is Good')
else: 
    print('Wine Quality is Bad')


# In[55]:


get_ipython().system('pip install streamlit')


# In[61]:


import streamlit as st


# In[62]:


import pickle


# In[63]:


filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[64]:


#loading the saved model
loaded_model = pickle.load(open(r'C:\Users\USER\trained_model.sav', 'rb'))


# In[65]:


data = (4.2,0.215,0.23,1.808289,0.041,64.0,157.0,0.99688,3.42,0.44,8.0)
array = np.asarray(data)
final = array.reshape(1,-1)
prediction = loaded_model.predict(final)
if (prediction[0] == 1):
    print('Wine Quality is Good')
else: 
    print('Wine Quality is Bad')

