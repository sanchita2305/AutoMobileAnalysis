#!/usr/bin/env python
# coding: utf-8

# # Automobile Data Wrangling

# 1.1 Objectives
# • Handle missing values
# • Correct data formatting
# • Standardize and normalize data

# Table of Contents
# Identify and handle missing values * Identify missing values * Deal with missing values Correct
# data format Data Standardization Data Normalization Binning Indicator Variable

# What is the purpose of data wrangling

# You use data wrangling to convert data from an initial format to a format that may be better for
# analysis

# What is the fuel consumption (L/100k) rate for the diesel car?
# Import data
# You can find the “Automobile Dataset” from the following link:
# https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data. You will
# be using this data set throughout this course.
# 

# Import pandas
# 

# In[3]:


#install specific version of libraries used in lab
#! mamba install pandas==1.3.3
#! mamba install numpy=1.21.2


# In[9]:


import pandas as pd
import matplotlib.pylab as plt


# In[11]:


file_name=r"C:\Users\Sai\Desktop\usedCars.csv"


# In[20]:


headers = ["symboling","normalized-losses","make","fuel-type","aspiration",
"num-of-doors","body-style",
"drive-wheels","engine-location","wheel-base",
"length","width","height","curb-weight","engine-type",
"num-of-cylinders",
"engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
"peak-rpm","city-mpg","highway-mpg","price"]


# Use the Pandas method read_csv() to load the data

# In[19]:


df = pd.read_csv(file_name, names = headers)


# Use the method head() to display the first five rows of the dataframe.
# 

# In[22]:


df.head()


# As you can see, several question marks appeared in the data frame; those missing values may hinder
# further analysis.
# So, how do we identify all those missing values and deal with them?
# How to work with missing data?

# Steps for working with missing data:

# 1.Identify missing data

# 2.Deal with missing data
# 

# 3.Correct data format
# 

# # 2. Identify and handle missing values
# 

# In[23]:


import numpy as np


# In[24]:


# replace "?" to NaN
df.replace("?", np.nan, inplace = True)
df.head(5)


# Evaluating for Missing Data
# 

# The missing values are converted by default.

# Use the following functions to identify these missing
# values. You can use two methods to detect missing data:

# 1. .isnull()
# 2. .notnull()
# 

# The output is a boolean value indicating whether the value that is passed into the argument is in
# fact missing data.

# In[26]:


missing_data = df.isnull()
missing_data.head(5)


# “True” means the value is a missing value while “False” means the value is not a missing value.
# 

# Count missing values in each column

# Using a for loop in Python, you can quickly figure out the number of missing values in each column.
# As mentioned above, “True” represents a missing value and “False” means the value is present in
# the data set. In the body of the for loop the method “.value_counts()” counts the number of “True”
# values.

# In[28]:


for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")
  


# 

# Based on the summary above, each column has 205 rows of data and seven of the columns containing
# missing data:

# “normalized-losses”: 41 missing data
# “num-of-doors”: 2 missing data
# “bore”: 4 missing data
# “stroke” : 4 missing data
# 8
# “horsepower”: 2 missing data
# “peak-rpm”: 2 missing data
# “price”: 4 missing data

# # Deal with missing data

# How should you deal with missing data?
# 

# Calculate the mean value for the “normalized-losses” column
# 

# In[29]:


avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)


# Replace “NaN” with mean value in “normalized-losses” column
# 

# In[30]:


df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)


# Calculate the mean value for the “bore” column
# 

# In[32]:


avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)


# Replace “NaN” with the mean value in the “bore” column
# 

# In[33]:


df["bore"].replace(np.nan, avg_bore, inplace=True)


# Question 1:
# Based on the example above, replace NaN in “stroke” column with the mean value.

# In[34]:


#Calculate the mean vaule for "stroke" column
avg_stroke = df["stroke"].astype("float").mean(axis = 0)
print("Average of stroke:", avg_stroke)

# replace NaN by mean value in "stroke" column
df["stroke"].replace(np.nan, avg_stroke, inplace = True)


# Calculate the mean value for the “horsepower” column
# 

# In[35]:


avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)


# Replace “NaN” with the mean value in the “horsepower” column

# In[36]:


df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)


# Calculate the mean value for “peak-rpm” column
# 

# In[37]:


avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)


# Replace “NaN” with the mean value in the “peak-rpm” column

# In[38]:


df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)


# To see which values are present in a particular column, we can use the “.value_counts()” method:

# In[39]:


df['num-of-doors'].value_counts()


# You can see that four doors is the most common type. We can also use the “.idxmax()” method to
# calculate the most common type automatically:

# In[41]:


df['num-of-doors'].value_counts().idxmax()


# The replacement procedure is very similar to what you have seen previously:
# 

# In[42]:


#replace the missing 'num-of-doors' values by the most frequent
df["num-of-doors"].replace(np.nan, "four", inplace=True)


# Finally, drop all rows that do not have price data:
# 

# In[43]:


# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)

# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)


# In[44]:


df.head()


# Correct data format

# The last step in data cleaning is checking and making sure that all data is in the correct format
# (int, float, text or other).
# In Pandas, you use:
# .dtype() to check the data type
# .astype() to change the data type

# Let’s list the data types for each column
# 

# In[45]:


df.dtypes


# Convert data types to proper format
# 

# In[46]:


df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")


# list the columns after the conversion

# In[47]:


df.dtypes


# # Data Standardization

# What is standardization?
# Standardization is the process of transforming data into a common format, allowing the researcher
# to make the meaningful comparison.

# In[48]:


df.head()


# In[49]:


# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]

# check your transformed data
df.head()


# In[50]:


# transform mpg to L/100km by mathematical operation (235 divided by mpg)
df["highway-mpg"] = 235/df["highway-mpg"]

# rename column name from "highway-mpg" to "highway-L/100km"
df.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)

# check your transformed data
df.head()


# # Data Normalization

# Why normalization?
# Normalization is the process of transforming values of several variables into a similar range. Typical
# normalizations include:
#     
# scaling the variable so the variable average is 0
# 
# scaling the variable so the variance is 1
# 
# scaling the variable so the variable values range from 0 to 1
# 

# Example
# To demonstrate normalization, say you want to scale the columns “length”, “width” and “height”.
# 

# Target: normalize those variables so their value ranges from 0 to 1

# Approach: replace the original value by (original value)/(maximum value)
# 

# In[51]:


# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()

df['width'] = df['width']/df['width'].max()


# Question #3:
# According to the example above, normalize the column “height”

# In[52]:


df['height'] = df['height']/df['height'].max()

# show the scaled columns
df[["length","width","height"]].head()


# # Binning

# Why binning?
# Binning is a process of transforming continuous numerical variables into discrete categorical ‘bins’
# for grouped analysis

# Example of Binning Data In Pandas

# Convert data to correct format:
# 

# In[53]:


df["horsepower"]=df["horsepower"].astype(int, copy=True)


# Plot the histogram of horsepower to see the distribution of horsepower.
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# Find 3 bins of equal size bandwidth by using Numpy’s linspace(start_value, end_value, numbers_generated function.
# Since you want to include the minimum value of horsepower, set start_value =
# min(df[“horsepower”]).
# Since you want to include the maximum value of horsepower, set end_value =
# max(df[“horsepower”]).
# Since you are building 3 bins of equal length, you need 4 dividers, so numbers_generated = 4.

# Build a bin array with a minimum value to a maximum value by using the bandwidth calculated
# above. The values will determine when one bin ends and another begins

# In[55]:


bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins


# Set group names:
# 

# In[57]:


group_names = ['Low', 'Medium', 'High']


# Apply the function “cut” to determine what each value of df['horsepower'] belongs to.
# 

# In[59]:


df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names,
include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)


# See the number of vehicles in each bin:
# 

# In[60]:


df["horsepower-binned"].value_counts()


# Plot the distribution of each bin:
# 

# In[61]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# In[62]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot

# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# # Indicator Variable

# What is an indicator variable?

# An indicator variable (or dummy variable) is a numerical variable used to label categories. They
# are called ‘dummies’ because the numbers themselves don’t have inherent meaning

# Why use indicator variables?
# 

# You use indicator variables so you can use categorical variables for regression analysis in the later
# modules.

# Use the Panda method ‘get_dummies’ to assign numerical values to different categories of fuel
# type.

# In[63]:


df.columns


# Get the indicator variables and assign it to data frame “dummy_variable_1”:
# 

# In[64]:


dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()


# Change the column names for clarity:

# In[66]:


dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':
'fuel-type-diesel'}, inplace=True)
dummy_variable_1.head()


# In the data frame, column ‘fuel-type’ now has values for ‘gas’ and ‘diesel’ as 0s and 1s.
# 

# In[67]:


# merge data frame "df" and "dummy_variable_1"
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)


# In[69]:


df.head()


# Question #4:
# Similar to before, create an indicator variable for the column “aspiration”

# In[70]:


# get indicator variables of aspiration and assign it to data frame "dummy_variable_2"
dummy_variable_2 = pd.get_dummies(df['aspiration'])

# change column names for clarity
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo':
'aspiration-turbo'}, inplace=True)

# show first 5 instances of data frame "dummy_variable_1"
dummy_variable_2.head()


# Question #5:
# Merge the new dataframe to the original dataframe, then drop the column ‘aspiration’

# In[71]:


# merge the new dataframe to the original datafram
df = pd.concat([df, dummy_variable_2], axis=1)

# drop original column "aspiration" from "df"
df.drop('aspiration', axis = 1, inplace=True)


# Save the new csv:
# 

# In[72]:


df.to_csv('usedCars.csv')


# In[ ]:




