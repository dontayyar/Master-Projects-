#!/usr/bin/env python
# coding: utf-8

# In[214]:


#my client company = Toykan Real Estate ınc.


# In[215]:


#Objective :
    # Enable Predictive Analysis on housing  prices according to area measurements. 
    # Train our dataset by creating end-to-end supervised machine learning algorithm.

    #Importance of this business:
    #- client company could understand and observe this data on  price marketing distribution according to area based features. and can offer customers the optimized value based home.
    


# In[216]:


#Steps: 
# 1. Load libraries for datasets. 
# 2. Data Preprocessing ( cleaning null features,dropping, logarithmic transformation, statistics).
# 3. Train and test our target values. 
# 4. Linear regression analysis for our trained data.
# 5. Array shaping of our target trained datas using numpy.
# 6. Line and Scatter Plotting using Pandas. 
# 7. Performance measure of our supervised learning by using MSE and RMSE. 
# 8. Creating a Decision tree model for increase  performance. 
# 9. Random forest algorithm used for higher performance
# 9. Data Visualization using Seaborn.


    


# In[217]:


#İmport libraries for to read data 
import pandas as pd 
import numpy as np 


# In[218]:


#read our dataset   
df = pd.read_csv(r"C:\Users\Toykan95\OneDrive\Masaüstü\teaching-main\teaching-main\datasets\housing.csv")
print(df)
df.head(60)


# In[219]:


df.shape #shaping our data 


# In[244]:


#drop unnecessary columns 
cols = ["id","stories.1"]

df = df.drop(cols,axis=1)


# In[245]:


df.info() #getting information of features with dropped columns 


# In[250]:


#statistics of data 
df.describe()


# In[251]:


#Cleaning and find  missing values using Pandas
df.dropna()


# In[252]:


# checking for missing values
df.isnull()  


# In[253]:


#no missing values 
df.isnull().sum() 


# In[254]:


df.duplicated().sum() #checking for duplicates


# In[255]:


#Logarithmic transformation for processing data using numpy 
import numpy as np 
import matplotlib.pyplot as plt 


np.random.seed(0)

#describing beta 
df = np.random.beta(a=3, b=10, size= 200)

#logarithmic transformation of our dataset
df_log = np.log(df)

#figure implementation
figure,axs = plt.subplots(nrows=1, ncols=2)


#Define hıstogram of original and transformed data 
axs[0].hist(df, edgecolor="red")
axs[1].hist(df_log, edgecolor="blue")

#set title 
axs[0].set_title("Original data")
axs[1].set_title("Log-transformed data")


# In[256]:


#Line Plotting using matplotlib
import pandas as pd 
from matplotlib import pyplot as plt
df = pd.read_csv(r"C:\Users\Toykan95\OneDrive\Masaüstü\teaching-main\teaching-main\datasets\housing.csv")

print(df.head())
df.plot.line(x="price", y="area")


# In[257]:


#Scatter Plotting using pandas 

#to see dot based distribution and correlation between price and bathroom 
df.plot(
   x='price', 
   y='area', 
   kind='scatter')

plt.show()


# In[258]:


#train and test model using sklearn 

#import sklearn libraries 
from sklearn.model_selection import train_test_split
X = df["price"]
y = df["area"]
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 10, train_size = 10, random_state = 42, shuffle=True, stratify=None)

#print our train and test values
print("X_train:")
print(X_train.head())
print("X_test:")
print(X_test.head())
print("y_train:")
print(y_train.head())
print("y_test:")
print(y_test.head())


# In[259]:


#using Linear Regression for training our data
import numpy as np
from sklearn.linear_model import LinearRegression

#numpy dimension 
price = df["price"]
area = df["area"]
X = np.array(price).reshape(-1,1) #array shaping using numpy
y = np.array(bathrooms)

#split train and test the dataset 
X_train,X_test,y_train,y_test= train_test_split(X, y, test_size = 0.2, random_state= 10)
y_train = y_train.reshape(-1,1) #array shaping Dimension
y_test = y_test.reshape(-1,1)   #array shaping Dimension


#Fitting our model 
my_model = LinearRegression()
my_model.fit(X_train,y_train)


#performance measure using Root Mean Square Error 
from sklearn import metrics
rmse = np.sqrt(metrics.mean_squared_error(y_train,my_model.predict(X_train)))
print(rmse)


# In[247]:


#Linear Regression based scatter plotting for visualize  the interception between  price and area  using matplot
import matplotlib.pyplot as plt
plt.scatter(X_train, y_train, color = 'green') 
plt.scatter(X_train, my_model.predict(X_train), color = 'red') 
plt.show() 


# In[248]:


#Decision Tree Regression by fitting our training data
from sklearn.tree import DecisionTreeRegressor
my_model = DecisionTreeRegressor(random_state=42)
my_model.fit(X_train, y_train)
predictions = my_model.predict(X_test)
print(predictions)


# In[233]:


#Classifying our decision tree
from sklearn.tree import DecisionTreeClassifier
my_model = DecisionTreeClassifier(random_state=10)
my_model.fit(X_train, y_train)


# In[249]:


#Predicting first 10 values 
predictions = dtc.predict(X_test)
print(predictions[:10])


# In[235]:


#Decision Tree model performance metrics 
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))


# In[236]:


#Randomforest classifying used for see higher performance by fitting our trained data 
from sklearn.ensemble import RandomForestClassifier
my_model = RandomForestClassifier(random_state=10)
my_model.fit(X_train,y_train.ravel()) #array shaping dimension in RandomForest algorithm, ravel is used 


# In[237]:


#Predicting our first 20 values 
predictions = my_model.predict(X_test)
print(predictions[:20])


# In[238]:


#Test metrics of RandomForest Model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))


# In[239]:


# data visualization using seaborn 
# Bathrooms between 3-2 have higher prices. 
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv(r"C:\Users\Toykan95\OneDrive\Masaüstü\teaching-main\teaching-main\datasets\housing.csv")
sns.scatterplot(x=data.price,y=data.bathrooms)


# In[240]:


#compare price differentiations  according to  bedrooms features by visualizing using seaborn 

sns.scatterplot(x=data.price,y=data.bedrooms)


# In[241]:


sns.scatterplot(x=data.price,y=data.furnishingstatus)


# In[242]:


#Results:
# In between three ML algorithms DecisionTree Model gives the highest accuracy score that fits our predictive analysis.
# price area correlation helps us most of the values are centralized at the lowest parts. 
# Estimated results shows us that bathroom selection have close relation within price 
# Analyze price changes in different features, especially for bedrooms and bathrooms. 


#Limitations:
#Hard to intrepret randomforest regression  into visualize ( memory is not enough)
#Decision tree plotting looks crowded, I had to remove it.
#array shaping in random forest classifier is take some time to understand and imply.


#Strengths: 
# Linear regression model structure is clearly gives result and a easy to create interception graph 
# Decision tree model gives the highest result which makes ou analysis stronger. 
# Decision tree and random forest tree algorithms preferable both because of similar accuracy scores.


#Data Driven recommendations:
#Client company can create a comparison chart on how area would effects the price within bedroom and bathroom numbers 
#Client company should divide customers according to their preferences, should make a priority chart according to my analysis.
#Client company could predict the price range of bedroom numbers and bathroom numbers


# In[243]:


#References:
#Stack Overflow. (n.d.). linear regression - How to get accuracy in RandomForest Model in Python? [online] Available at: https://stackoverflow.com/questions/55942884/how-to-get-accuracy-in-randomforest-model-in-python.
#GeeksforGeeks. (2017). Python | Decision tree implementation. [online] Available at: https://www.geeksforgeeks.org/decision-tree-implementation-python/?ref=gcse [Accessed 23 Mar. 2023].
#GeeksforGeeks. (2019). Random Forest Regression in Python. [online] Available at: https://www.geeksforgeeks.org/random-forest-regression-in-python/?ref=rp [Accessed 23 Mar. 2023].
#www.learndatasci.com. (n.d.). Intro to Feature Engineering for Machine Learning with Python. [online] Available at: https://www.learndatasci.com/tutorials/intro-feature-engineering-machine-learning-python/.

