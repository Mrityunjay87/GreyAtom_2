# -*- coding: utf-8 -*-
"""
Created on Sat May 16 04:34:07 2020

@author: Mrityunjay1.Pandey
"""

path='./iowa_housing.csv'

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 

# Code starts here
#Reading data using pandas
ames=pd.read_csv(path)
#splitting data in features and target

#Features
X=ames.drop('SalePrice',axis=1)
#Target
y=ames.SalePrice
#Splitting training and test data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

X_train['Class']=y_train

#Finding Correlation and keeping the value related to target
t_corr=X_train.corr().Class

#Slicing features whose correlation is greater than o.5
corr_columns=t_corr[abs(t_corr)>0.5]
#Removing class column
corr_columns.drop('Class',axis=0,inplace=True)

#Subsetting datframe to keep relevent columns according to the correaltion data
X_train_new=X_train[corr_columns.index]

X_test_new=X_test[corr_columns.index]

#Intialising linear regression model
model=LinearRegression()
model.fit(X_train_new,y_train)
y_pred=model.predict(X_test_new)

#Finding r^2 score
corr_score=model.score(X_test_new,y_test)

# Code ends here

#Chi Squared Test
# Code starts here

#Features
X=ames.drop('SalePrice',axis=1)
#Target
y=ames.SalePrice
#Initialisation of Chi squared
test=SelectKBest(score_func=chi2,k=50)
#Splitting training and test data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
X_train=test.fit_transform(X_train,y_train)
X_test=test.transform(X_test)

#Intialising linear regression model
model=LinearRegression()
model.fit(X_train,y_train)
y_pred_chi=model.predict(X_test)

#Finding Chi2 score
chi2_score=model.score(X_test,y_test)

#ANOVA Test
# Code starts here

#Features
X=ames.drop(['Id','SalePrice'],axis=1)
#Target
y=ames.SalePrice
#Initialisation of Chi squared
test=SelectKBest(score_func=f_regression,k=60)
#Splitting training and test data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
X_train=test.fit_transform(X_train,y_train)
X_test=test.transform(X_test)

#Intialising linear regression model
model=LinearRegression()
model.fit(X_train,y_train)
y_pred_chi=model.predict(X_test)

#Finding Chi2 score
f_regress_score=model.score(X_test,y_test)


#Wrapper methods
# Feature Extraction with RFE

#no of features list
nof_list=[20,30,40,50,60,70,80]

#Variable to store the highest score
high_score=0

#Variable to store the optimum features
nof=0

#Code begins here

#Features
X=ames.drop('SalePrice',axis=1)
#Target
y=ames.SalePrice
#Initialisation of Chi squared
for n in nof_list:    
    #Splitting training and test data
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
    #Model Initialisation
    model=LinearRegression()
    rfe=RFE(model,n)
    X_train_rfe=rfe.fit_transform(X_train,y_train)
    X_test_rfe=rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    r2_score=model.score(X_test_rfe,y_test)
    if r2_score>high_score:
        high_score=r2_score
        nof=n
print("High Score is {} with no of Features {}".format(high_score,nof))


#Embedded methods
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

#Lasso Model
lasso=Lasso(random_state=0)
lasso.fit(X_train,y_train)
lasso_score=lasso.score(X_test,y_test)

#Ridge Model
ridge=Ridge(random_state=0)
ridge.fit(X_train,y_train)
ridge_score=ridge.score(X_test,y_test)



#PCA

#Features
X=ames.drop('SalePrice',axis=1)
#Target
y=ames.SalePrice
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
pca=PCA(n_components=30,random_state=0)

X_train_pca=pca.fit_transform(X_train_scaled)
X_test_pca=pca.transform(X_test_scaled)

#Intialising linear regression model
model=LinearRegression()
model.fit(X_train_pca,y_train)

pca_score=model.score(X_test_pca,y_test)

print("Lasso Score is: {}\n Ridge Score is:{}\nPCA Score is:{}".format(
        lasso_score,ridge_score,pca_score))

