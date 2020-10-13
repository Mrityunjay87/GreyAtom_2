# -*- coding: utf-8 -*-
"""
Created on Mon May  4 11:20:42 2020

@author: Mrityunjay1.Pandey
"""

#Task 1 -  Data Loading
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Code starts here
data=pd.read_csv(path)

data.Rating.hist(bins=10)
plt.title("Histogram Without cleaning")

data=data[data.Rating<=5]

plt.hist(data.Rating)
plt.title("Histogram after Slicing")

plt.show()
#Code ends here

#Task 2 - Null Value Treatment
# code starts here

total_null=data.isnull().sum()

percent_null=(data.isnull().sum()/data.isnull().count())
missing_data=pd.concat((total_null,percent_null),axis=1,keys=['Total','Percent'])
print("Missing Data is:",missing_data)

data=data.dropna()
total_null_1=data.isnull().sum()

percent_null_1=(data.isnull().sum()/data.isnull().count())
missing_data_1=pd.concat((total_null_1,percent_null_1),axis=1,keys=['Total','Percent'])

print("Missing Data counts in columns is:",missing_data_1)


# code ends here

#Task 3 - Category vs Rating

#Code starts here

ax=sns.catplot(x="Category",y="Rating",data=data, kind="box",height = 10)
[plt.setp(ax.get_xticklabels(), rotation=90) for ax in ax.axes.flat]
ax.fig.suptitle('Rating vs Category [BoxPlot]')

#Code ends here

#Task 4
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
print(data.Installs.value_counts())
data['Installs']=data['Installs'].str.replace('+','')
data['Installs']=data['Installs'].str.replace(',','')
data['Installs']=data['Installs'].astype(int)

print(data.Installs.value_counts())
le=LabelEncoder()
le.fit_transform(data.Installs)

g=sns.regplot(x='Installs',y='Rating',data=data,)
g.set(title='Rating vs Installs [RegPlot]')
plt.show()


#Code ends here
