# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 23:35:21 2020

@author: Mrityunjay1.Pandey
"""

path='./EDA.csv'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

#Task 1
df=pd.read_csv(path)
#Splitting into features and target
X=df.iloc[:,:7]
y=df.iloc[:,7]
#Splitting the data in train and test
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=2)

print("\t\tTraining Data is:-\n{}\n\t\tTest Data is:-\n{}".format(X_train,X_test))

#Task 2
plt.figure(figsize=(10,12))

fig, ((ax_1,ax_2),(ax_3,ax_4)) = plt.subplots(2,2)

fig.tight_layout(pad=2.5)

ax_1.scatter(X_train['LotFrontage'],y_train,color='red')
ax_1.set_title("SalePrice vs LotFrontage")
ax_1.set_xlabel('LotFrontage')
ax_1.set_ylabel('SalePrice')

ax_2.scatter(X_train['TotalBsmtSF'],y_train,color='blue')
ax_2.set_title("SalePrice vs TotalBsmtSF")
ax_2.set_xlabel('TotalBsmtSF')
ax_2.set_ylabel('SalePrice')

ax_3.scatter(X_train['GrLivArea'],y_train,color='green')
ax_3.set_title("SalePrice vs GrLivArea")
ax_3.set_xlabel('GrLivArea')
ax_3.set_ylabel('SalePrice')

ax_4.scatter(X_train['LotArea'],y_train,color='black')
ax_4.set_title("SalePrice vs LotArea")
ax_4.set_xlabel('LotArea')
ax_4.set_ylabel('SalePrice')

plt.show()

#Task 3
train=pd.concat([X_train,y_train],axis=1)
mask1=train.LotFrontage<300

mask2=train.TotalBsmtSF<5000
mask3=train.GrLivArea<4500
mask4=train.LotArea<100000

train=train[mask1 & mask2 & mask3 & mask4]

#Task4
X_train, y_train = train.iloc[:,:7], train[['SalePrice']]

missing_columns=(X_train.isnull().sum()/len(X_train))*100

mask=missing_columns>50

columns=missing_columns[mask].index.tolist()
print("Columns having more than 50% data missing are:{}".format(columns))
rows_percentage= (1 - (len(X_train.dropna(thresh=5)) / len(X_train)))*100
print("Row percentag with more than 5 rows missing is:{}".format(rows_percentage))

#Task 5
# Import packages
from sklearn.preprocessing import Imputer
dict_new = {'Attchd':0,'Detchd':1,'BuiltIn':2,'2Types':3,'CarPort':4,'Basment':5}
X_train['GarageType'] = X_train['GarageType'].map(dict_new)
X_test['GarageType'] = X_test['GarageType'].map(dict_new)

#Task 6
# Custom imputers
mean_imputer = Imputer(strategy='mean')
mode_imputer = Imputer(strategy='most_frequent')
 
X_train.drop('PoolQC',axis=1,inplace=True)
X_test.drop('PoolQC',axis=1,inplace=True)

mode_imputer=mode_imputer.fit(X_train[['GarageType']])
X_train[['GarageType']] = mode_imputer.transform(X_train[['GarageType']])
X_test[['GarageType']]=mode_imputer.transform(X_test[['GarageType']])

mean_imputer=mean_imputer.fit(X_train[['LotFrontage']])
X_train[['LotFrontage']] = mean_imputer.transform(X_train[['LotFrontage']])
X_test[['LotFrontage']]=mean_imputer.transform(X_test[['LotFrontage']])

#Task 7
y_train=np.log(y_train)
sns.distplot(y_train)

#Task 8
# numerical columns
num_columns = ['LotFrontage', 'TotalBsmtSF', 'GrLivArea', 'LotArea']

# Import packages
from sklearn.preprocessing import MinMaxScaler
normalizer=MinMaxScaler()
normalizer.fit(X_train[num_columns])
X_train[num_columns] = normalizer.transform(X_train[num_columns])
X_test[num_columns] = normalizer.transform(X_test[num_columns])
print("After scaling with MinMaxScaler\n",X_train.head(5))
print("With Scaling\n",X_test.head(5))

#Task 9
# Convert to category type using type casting
X_train['SaleCondition'] = X_train['SaleCondition'].astype('category')

# Label encode 'SaleCondition' feature
X_train['SaleCondition'] = X_train['SaleCondition'].cat.codes

# Look at the first five rows
print(X_train.head())

"""sklearn method to encoding categorical data
# Import packages
from sklearn.preprocessing import LabelEncoder

# Initialize encoder object
encoder = LabelEncoder()

# Fit-transform on data
X_train['SaleCondition'] = encoder.fit_transform(X_train['SaleCondition'])
"""
from sklearn.preprocessing import LabelEncoder
#Label Encoding
label_encoder=LabelEncoder()
label_encoder.fit_transform(X_train.SaleCondition)
label_encoder.fit_transform(X_test.SaleCondition)

#One hot encoding on GarageType 
x_train = pd.get_dummies(X_train['GarageType'])
x_test= pd.get_dummies(X_test['GarageType'])

print(x_train.head(5))
print(x_test.head(5))
