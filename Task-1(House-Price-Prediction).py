#!/usr/bin/env python
# coding: utf-8

# ## Import the packages and the dataset

# In[43]:


#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[7]:


#import the dataset
house_price=pd.read_csv('kc_house_data.csv')


# In[8]:


house_price


# ## Data Preprocessing

# In[9]:


house_price.shape


# In[10]:


house_price.head(10)


# In[11]:


house_price.tail(10)


# In[15]:


house_price.describe()


# In[16]:


#checking for null values
house_price.info()


# In[20]:


# Handle missing values
house_price.fillna(method='ffill', inplace=True)  # Forward fill missing values


# In[18]:


### check null entries
house_price.isnull().sum()


# In[21]:


## remove duplicate entries
house_price.drop_duplicates(inplace = True)
house_price


# In[22]:


# Replace missing values in numerical columns with mean or median
for col in house_price.columns:
    if house_price[col].dtype == 'float64' or house_price[col].dtype == 'int64':
        house_price[col].fillna(house_price[col].mean(), inplace=True)

# Replace missing values in categorical columns with most frequent category
for col in house_price.columns:
    if house_price[col].dtype == 'object':
        house_price[col].fillna(house_price[col].mode()[0], inplace=True)


# In[24]:


## Standardize numerical features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
house_price[['sqft_living', 'bedrooms', 'bathrooms']] = scaler.fit_transform(house_price[['sqft_living', 'bedrooms', 'bathrooms']])


# ## Data Exploration

# In[ ]:





# In[27]:


house_price.hist()


# In[25]:


sns.histplot(house_price['price'],bins = 20)


# In[26]:


sns.histplot(house_price['bedrooms'],bins=20)


# In[28]:


house_price["bedrooms"].value_counts().plot(kind='bar')
plt.title('Count of bedrooms')


# In[29]:


house_price["bathrooms"].value_counts().plot(kind='bar')
plt.title('Count of bathrooms')


# In[33]:


plt.figure(figsize=(10,6))
sns.barplot(data = house_price, x = 'bathrooms', y = 'price');
plt.show()


# In[36]:


#Feature Selection/Engineering
# Select relevant features or create new features based on your dataset
X = house_price[['sqft_living', 'bedrooms', 'bathrooms']]  # Features
y = house_price['price']  # Target variable


# In[38]:


# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Train the model

# In[40]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[44]:


#Evaluate the model's performance on the test set

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', mse)


# In[47]:


#Use the trained model to predict the price of a new house

new_house = [[1500, 3, 2]]  # Features of a new house to predict its price
predicted_price = model.predict(new_house)
print(f"Predicted Price: {predicted_price}")

'''new_house_data = {'sqft_living': 1500, 'number_of_bedrooms': 3}
new_house_X = pd.DataFrame([new_house_data])

predicted_price = model.predict(new_house_X)
print('Predicted Price:', predicted_price[0])''' 

