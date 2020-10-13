# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 00:06:51 2020

@author: Mrityunjay1.Pandey
"""

feature_path='./Data_Subset_bank_marketting.csv'
label_path='./Data_Subset_bank_marketting_label.csv'

import pandas as pd
#Reading Files
bank_full=pd.read_csv(feature_path)
bank_full_labels=pd.read_csv(label_path)

#Viewing top 5 rows of the data
print("Bank Full Data:\n",bank_full.head(5))
print("Bank Full Label Data:\n",bank_full_labels.head(5))



