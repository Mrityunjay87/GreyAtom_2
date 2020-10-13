# -*- coding: utf-8 -*-
"""
Created on Fri May  1 09:10:36 2020

@author: Mrityunjay1.Pandey
"""
path='./PlayStore_Data.csv'

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
data=pd.read_csv(path)
#Task 1
data.Rating.hist(bins=10)
plt.title("Histogram Without cleaning")


data=data[data.Rating<=5]

data.Rating.hist(bins=10)
plt.title("Histogram after Slicing")

plt.show()

#Task 2

total_null=data.isnull().sum()

percent_null=(data.isnull().sum()/data.isnull().count())
missing_data=pd.concat((total_null,percent_null),axis=1,keys=['Total','Percent'])
print("Missing Data is:",missing_data)

data=data.dropna()
total_null_1=data.isnull().sum()

percent_null_1=(data.isnull().sum()/data.isnull().count())
missing_data_1=pd.concat((total_null_1,percent_null_1),axis=1,keys=['Total','Percent'])

print("Missing Data counts in columns is:",missing_data_1)

#Task 3
ax=sns.catplot(x="Category",y="Rating",data=data, kind="box",height = 10)
[plt.setp(ax.get_xticklabels(), rotation=90) for ax in ax.axes.flat]
ax.fig.suptitle('Rating vs Category [BoxPlot]')
plt.show()

#Task 4
print(data.Installs.value_counts())
#Original Attempt
data['Installs']=data.Installs.str.replace("[,+]","",regex=True).astype(int)

#As suggested by GLabs hint.

le=LabelEncoder()
data.Installs=le.fit_transform(data.Installs)

g=sns.regplot(x='Installs',y='Rating',data=data,)
g.set(title='Rating vs Installs [RegPlot]')
print(data.Installs.value_counts())
plt.show()


#Task 5

print(data.Price.value_counts())

data.Price=data.Price.str.replace("$","").astype(float)

sns.regplot(x="Price",y="Rating",data=data)
plt.title('Rating vs Price [RegPlot]')
plt.show()

#Task 6
data=data.reset_index(drop=True)
data.Genres=data.Genres.str.split(";").str[0]


gr_mean=data[['Genres','Rating']].groupby(by='Genres',as_index=False).mean()

gr_mean.describe()
gr_mean=gr_mean.sort_values('Rating')

print("First Value is:\n{}\n and Last Value is:\n{} ".format(gr_mean.iloc[0],gr_mean.iloc[len(gr_mean)-1]))

#Task 7
data['Last Updated']=pd.to_datetime(data['Last Updated'])
max_date=data['Last Updated'].max()
data['Last Updated Days']=(max_date-data['Last Updated']).dt.days

sns.regplot(x="Last Updated Days",y="Rating", data=data)
plt.title("Rating vs Last Updated [RegPlot]")
plt.show()


    
    
    
    
    