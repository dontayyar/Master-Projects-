#!/usr/bin/env python
# coding: utf-8

# In[4]:


#My Client Company : Toykan's Car Services Inc.


# In[5]:


# Business Questions:
#What is the most produced company model ?
#Which company has the highest horsepower  ? 
#According to qualitative features which body type is highly produced ?
#Which engine type has the highest price in which company model ? 
#Which body style has the lowest price values in which company model ? 
#What is the most prefered engine type in between 160 -180 length of models?
#Which company model has the highest price in between 100-150 horsepower ? 
#In which range of length and price has most company models ?
#Which body style has most common company models in between price of 1000-2000?
#In which  range of wheel base has company models produce ( mostly centered )? What is the price range ?  


# In[6]:


#Answers:
#Alfa Romeo
#Porsche
#Sedan
#ohcf 
#toyota hatchback
#ohc
#mercedes benz
#170-180 meter and 500-1000 price has most company models
#Sedan 
#90-100 wheel base is mostly prefered 


# In[7]:


#import relevant libraries for data analysis
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# In[8]:


#reading our data by using pandas 
df= pd.read_csv(r"C:\Users\Toykan95\OneDrive\Masaüstü\teaching-main\teaching-main\datasets\automobile.csv")
print(df)


# In[9]:


#manipulating our data using pandas 


# In[10]:


df.head(61) #categorizing data more properly


# In[11]:


#featuring non-null values and memory usages according to our variables 
df.info() 


# In[12]:


# statistical description
df.describe() 


# In[13]:


df.dropna()
#dropping missing values if any


# In[14]:


df.isnull() #if any zero values checking 


# In[15]:


df.isnull() # NaN values or Zero values


# In[16]:


df.isnull().any() #no zero values except price 


# In[17]:


df.isnull().sum() #counting zero or NaN values 


# In[18]:


df.columns #listing variables


# In[19]:


df.duplicated().value_counts() #checking for duplication


# In[20]:


#using pie chart for efficient percentage showing for company models
type_count_auto = df["company"].value_counts()

#creating a pie chart using matplot
figure, ax = plt.subplots()
ax.pie(type_count_auto,labels=type_count_auto.index,autopct="%1.1f%%")

#representation
plt.show()

#setting title for our pie model
ax.set_title("Company Auto Model Distribution")


# In[21]:


# Demograph most common qualitative features in Histogram plotting


# In[22]:


plt.hist(df['engine-type'], edgecolor="white", color="purple")


# In[23]:


plt.hist(df['body-style'], edgecolor = "red", color="green")


# In[24]:


plt.hist(df['num-of-cylinders'], edgecolor="yellow", color="orange")


# In[25]:


#Representing Market values of Company models using matplot 


# In[50]:


comp_mean = df.groupby("company")["price"].mean(). sort_values(ascending=False)
plt.figure(figsize=(9,4))
comp_mean.plot(kind="barh", color="green")
plt.xlabel("price")
plt.ylabel("company")
plt.title("Price Values depend on models")


# In[27]:


#Representing Company models amount of  horsepower using matplot. 


# In[28]:


comp_mean = df.groupby("company")["horsepower"].mean(). sort_values(ascending=False)
plt.figure(figsize=(9,4))
comp_mean.plot(kind="barh", color="purple")
plt.xlabel("horsepower")
plt.ylabel("company")
plt.title("Horsepower scale  in  auto type")


# In[29]:


#Representing milleage  per each car model using matplot


# In[49]:



comp_mean = df.groupby("company")["average-mileage"].mean(). sort_values(ascending=False)
plt.figure(figsize=(9,4))
comp_mean.plot(kind="barh", color="brown")
plt.xlabel("average-mileage")
plt.ylabel("company")
plt.title("Average mileage according to   auto type")


# In[31]:


#Car Prices correlation according features: horsepower,engine_types,body styles,num-of-cylinders,length for customers preferences using Seaborn


# In[32]:


#try to help on  customers who  select their car according to horsepower, could make a price comparison.
sns.scatterplot(x="horsepower",y="price", hue="company", data=df,palette="Dark2", s=50)
plt.title("Price difference between horsepower  depends on company")


# In[33]:


#Helps to visualize color based scattering, attracts customers who is into engine types, could be make price comparison according to company models.
sns.scatterplot(x="price", y="engine-type", hue="company", data=df, palette="Dark2", s=70)
plt.title("Price difference between engine types  within  company")


# In[34]:


#Color based scattering for visualizing models for customers who wants to select his/her car model according to body style with price comparison. 
sns.scatterplot(x="price", y="body-style", hue="company", data=df, palette="Dark2", s=70)
plt.title("Price difference between body styles  within  company")


# In[35]:


#Helps to achieve customers who is make preferences on wheel base, could make price comparison before selecting a model. 
#Wheel base correlation in between price and company models using seaborn 
sns.scatterplot(x="price", y="wheel-base", data=df,hue="company", palette="Dark2",s=60)
plt.title("price versus wheel-base according to Company")


# In[36]:


#Help to achieve on customer who has  prior preferences in cyclinder amounts inserted in models, could be able to  make price comparison 
#Wheel-base correlation between prices within Company models using seaborn
sns.scatterplot(x="price", y="num-of-cylinders", data=df,hue="company", palette="Dark2",s=60)
plt.title("price versus num-of-cylinders  according to Company")


# In[37]:


#Helps to achieve on Customer who could prefer longer or shorther models especially depend on models and price. 
#Car length correlation between price within company models using seaborn
sns.scatterplot(x="length", y="price", data=df,hue="company", palette="Dark2",s=60)
plt.title("length relation between price according to company models")


# In[38]:


#Customer could search a model  according to mileages, with a price comparison
sns.scatterplot(x="average-mileage", y="price", data=df,hue="company", palette="Dark2",s=60)
plt.title("average mileage relation between price according to company models")


# In[39]:


#Understanding customers who prefers a specific engine-types with expected length in a  car  before buying
#Relation between car length and engine-type scatter plotting using matplot
plt.scatter(x="length",y="engine-type", data=df, linewidth=4)
plt.xlabel("length")
plt.ylabel("engine-type")
plt.title("length versus engine-type")


# In[40]:


#Observation of most common quantitative  features by Seaborn


# In[41]:


sns.distplot(df["average-mileage"],color="blue", label="sepal_width")
# most common mileage is in between 22-25.


# In[42]:


sns.distplot(df["length"],color="red", label="sepal_width")
# most common length is approximately 177 meter


# In[43]:


sns.distplot(df["wheel-base"], color="brown", label="sepal_width")
#most common  wheel perimeter is approximately 95. 


# In[44]:


sns.distplot(df["price"], color="green", label="sepal_width")
#most common price  is close to 10000 


# In[45]:


#Heatmapping for numerical features to observe strong or weak connections.


# In[46]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),cmap="coolwarm",annot=True)


# In[47]:


#Results and Conclusion: 
#Based upon my data analysis I reached that horsepower has affection on prices by observing the scatter plotting. 
#Customer can see insights of distribution of engine types, wheels, cyclinder,  length and average milleages body style features and have  a general knowledge before buying.
#customer could make a relation between wheel base and price and able to choose a optimized  model.
#According to our results that horsepower has most related   data with price increase.
#Mercedes Benz has the smallest average-milleage percentage with a highest price. 
#Chevrolet has the smallest smallest horsepower and lowest price could be convenient for family members 

#Importance of this data analysis: 
#client partners can compare the   average milleage  for each car model. 
#customer can able  to select most efficient engine types and body types in a company model by comparing their prices. 
#client company can suggest a model with qualitative features across accurate price.
#clients could be able to understand strong correlations and weak correlations across heatmapping.
#scatter plottings on price based features within company models will have high succession on which model could be suggest to a customer.
#cost efficiency based graphs 



#Strengths: 
# Easy to make both qualitative and quantitative analysis in histograms
# color scatter plotting is understandable for data readers
# heatmapping illustrates clear way of showing correlation matrices. 


#Limitations:
# hard to collobrate a company model preferences in between qualitative features.
# Observing scatter plotting with groupby  different variables hard to read.
# Not easy to plot and could not be able to  correlate in between non-numerical features.
# No customer side feedback  for  rating a car model or a feature.(additional columns on customer features) 
# Automobile ages could be added in order to compare  approximate lifetime according to their prices. 
# Price depends on many features,thats why seperately analyze for each features. Not an easy case. 
# Lack of information on company models (production year could be included)




#Data Driven Business Recommendation:
#Client company should expand their search on automobile firms within new technology features, modification types should be compared with my analysis  and  most popular and cost efficient features present to customers.
#Compare model features for competitive advantages and aim for cost  performance and optimized models.
#Data Speculation,cleaning is necessary for compare features and analyze for sales structure. 


# In[48]:


#REFERENCES:

#datascience, A. (2022). Exploratory Data Analysis in Python — A Step-by-Step Process. [online] Medium. Available at: https://towardsdatascience.com/exploratory-data-analysis-in-python-a-step-by-step-process-d0dfa6bf94ee.
#Python, R. (n.d.). Visualizing Data in Python Using plt.scatter() – Real Python. [online] realpython.com. Available at: https://realpython.com/visualizing-python-plt-scatter/.
#Nik (2020). Creating a Histogram with Python (Matplotlib, Pandas) • datagy. [online] datagy. Available at: https://datagy.io/histogram-python/.
#Stack Overflow. (n.d.). python - Use Pandas index in Plotly Express. [online] Available at: https://stackoverflow.com/questions/57178206/use-pandas-index-in-plotly-express [Accessed 21 Mar. 2023].




