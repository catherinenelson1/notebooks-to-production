#!/usr/bin/env python
# coding: utf-8

# ## Download dataset

# In[1]:


# dataset from https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv
import requests

url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/refs/heads/master/penguins.csv'
response = requests.get(url)

with open('penguins_data.csv', 'wb') as file:
    file.write(response.content)


# ## Explore and clean the data

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('penguins_data.csv')


# In[3]:


# len(df)


# In[4]:


# df.head()


# In[5]:


# df.isnull().sum()


# In[6]:


# drop rows with missing values
df = df.dropna(subset=['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])


# In[7]:


# len(df)


# In[8]:


# select only columns with relevant features
df = df[['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]


# In[9]:


# df.head()


# In[10]:


# df['species'].value_counts()


# In[11]:


features = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']].to_numpy()
values = df['species'].to_numpy()


# ## Scale and encode the data

# In[48]:


from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# In[13]:


# features


# In[61]:


scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# In[62]:


# features_scaled


# In[63]:


# one-hot encode the species
encoder = LabelEncoder()
values_encoded = encoder.fit_transform(values)


# In[64]:


# values_encoded


# In[65]:


# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, values_encoded, test_size=0.2, random_state=42)


# ## Try out some models

# In[66]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# In[78]:


lr = LogisticRegression()
lr.fit(X_train, y_train)



# In[79]:


y_pred_lr = lr.predict(X_test)


# In[80]:


# print(classification_report(y_test, y_pred_lr, target_names=['Adelie', 'Gentoo', 'Chinstrap']))


# In[83]:


# print(confusion_matrix(y_test, y_pred_lr))



# ### Make a prediction on new data

# In[70]:


scaled_data = scaler.transform([[40, 17, 190, 3500]])
prediction = lr.predict(scaled_data)



# In[86]:


print([encoder.inverse_transform(prediction)[0], lr.predict_proba(scaled_data)[0][prediction[0]]])




# In[85]:
scaled_data = scaler.transform([[80, 10, 250, 2500]])
prediction = lr.predict(scaled_data)


# In[86]:

print([encoder.inverse_transform(prediction)[0], lr.predict_proba(scaled_data)[0][prediction[0]]])

