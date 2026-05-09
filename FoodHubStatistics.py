# -*- coding: utf-8 -*-
"""
Created on Sat May  9 16:26:04 2026

@author: mvale
"""

# import libraries for data manipulation
import numpy as np
import pandas as pd

# import libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# read the data
df = pd.read_csv(r'G:\My Drive\MIT_ML\Week2\foodhub_order.csv')
# returns the first 5 rows
df.head()
df.info()

# Cost histogram
sns.histplot(data=df, x='cost_of_the_order', kde=True)

# Cost boxplot
sns.boxplot(data=df, x='cost_of_the_order')
df['cost_of_the_order'].median()

# Food prep time histogram
sns.histplot(data=df, x='food_preparation_time', kde=True)

# Food prep time boxplot
sns.boxplot(data=df, x='food_preparation_time')

# Delivery time histogram
sns.histplot(data=df, x='delivery_time', kde=True)

# Delivery time boxplot
sns.boxplot(data=df, x='delivery_time')

# Checking the value counts is more useful for understanding
#    which restaurants have the most orders
df['restaurant_name'].value_counts()

# Cuisine type countplot
sns.countplot(data=df, x='cuisine_type')
plt.xticks(rotation=90)
plt.show()

# Day of the week countplot
sns.countplot(data=df, x='day_of_the_week')

# Rating countplot, sorting by count
sns.countplot(data=df, x='rating', order=df['rating'].value_counts().index)

# looking for top 5 restaurants in # orders received
df['restaurant_name'].value_counts()

subset = df[df['day_of_the_week'] == 'Weekend'] #subset df by orders that occur on a weekend
subset['cuisine_type'].value_counts() #get counts of cuisine types

subset = df[df['cost_of_the_order'] > 20] #subset by orders that cost more than 20 dollars
len(subset) / len(df) * 100 

# mean delivery time
df['delivery_time'].mean()

# Pairplot to assess relationships between numeric variables
sns.pairplot(data=df, vars=['cost_of_the_order','food_preparation_time','delivery_time'])

columns = ['delivery_time', 'cost_of_the_order', 'food_preparation_time']
sns.heatmap(df[columns].corr(), annot=True)
plt.show()

# food prep time x cuisine type
sns.catplot(data=df, x='cuisine_type', y='food_preparation_time', kind='box')
plt.xticks(rotation=90)
plt.show()

# day of the week x delivery time
sns.violinplot(data=df, x='day_of_the_week', y='delivery_time')

# Cuisine type x cost
sns.catplot(data=df, x='cuisine_type', y='cost_of_the_order', kind='box')
plt.xticks(rotation=90)
plt.show()

# cost x rating
sns.catplot(data=df, x='rating', y='cost_of_the_order', kind='point')

# food prep time x rating
sns.catplot(data=df, x='rating', y='food_preparation_time', kind='point')

# delivery time x rating
sns.catplot(data=df, x='rating', y='delivery_time', kind='point')

# cost x day of the week
sns.catplot(data=df, x='day_of_the_week', y='cost_of_the_order', kind='violin')

# subset by observations with ratings, only
subset = df[df['rating'] != 'Not given'].copy() 

# get counts of ratings per restaurant
count_ratings = subset['restaurant_name'].value_counts()

# get restaurants with more than 50 ratings
names = count_ratings[count_ratings > 50]
names = list(names.index)

# get df of only restaurants with > 50 ratings
subset_final = subset[subset['restaurant_name'].isin(names)].copy()

# convert rating to int so we can calculate average
subset_final['rating'] = subset_final['rating'].astype('int')

# get means of restaurants
subset_final.groupby(['restaurant_name'])['rating'].mean()

#subset by cost > 20 dollars
df_20ormore = df[df['cost_of_the_order'] > 20].copy()

# calculate revenue from orders with 25 percent charge
sum_cost_20ormore = sum(df_20ormore['cost_of_the_order'] * .25)

# subset by 20 => cost > 5 
df_btw5n20 = df[(df['cost_of_the_order'] > 5) &
               (df['cost_of_the_order'] <= 20)].copy()

# calculate revenue from orders with 15 percent charge
sum_cost_btw5n20 = sum(df_btw5n20['cost_of_the_order'] * .15)

# sum revenues
sum_cost_20ormore + sum_cost_btw5n20

# get total number of orders for denominator
ct_all_orders = len(df)

# create column for summed prep time and delivery time
df['tot_order_time'] = df['delivery_time'] + df['food_preparation_time']

# get total number of observations where tot_order_time > 60
ct_60_min = len(df[df['tot_order_time'] > 60])

# calculate percentage
ct_60_min / ct_all_orders * 100

# calculate mean delivery time on weekends
df.groupby(['day_of_the_week'])['delivery_time'].mean()


# CONCLUSIONS AND RECOMMENDATIONS

# Conclusions:
# The median cost of orders is roughly 14 dollars, with the 25th percentile falling at approximatly 12 dollars, and the 75th percentile falling at approximately 23 dollars.
# American, Japanese, Italian, and Chinese cuisines are the most popular, and account for the most orders. These cuisines also have similar food prep times and costs.
# Weekday deliveries take 5.87 more minutes, on average, over weekend delivery times.
# Higher cost orders tend to have higher ratings.
# Longer delivery times are slightly more associated with lower ratings.
# Recommendations:
# Investigate time of day for weekend and weekday deliveries. It is possible that more concentrated delivery times during weekdays (e.g., after work) may lead to longer delivery times. Therefore, having more drivers available may lead to decreased delivery times, which may yield higher feedback ratings, and lead to more orders in the long-term.
# American, Japanese, Italian, and Chinese cuisines are the most popular orders. These cuisine types may be candidates for additional consumer incentives or promotions, potentially attracting more customers or retaining current customers for repeat orders.
# Consider adjusting the threshold for the 25% restaurant charge. Currently, the 20 dollar threshold sits just under the 75th percentile of the data, meaning that there is potential for increasing the order pool subject to 25% charge if the threshold is lowered slightly. However, FoodHub would need to consider the strain on businesses for this change.















