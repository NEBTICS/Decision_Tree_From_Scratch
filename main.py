# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 19:08:52 2022

@author: smith_barbose
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Decision_Tree import DecisionTree 

#%%
data=pd.read_csv("creditcard.csv")
sc = StandardScaler()
amount = data['Amount'].values
data['Amount'] = sc.fit_transform(amount.reshape(-1, 1))
data.drop(['Time'], axis=1, inplace=True)
data.drop_duplicates(inplace=True)
#%%

X = data.drop('Class', axis = 1).values
y = data['Class'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
#%%
def accuracy(y_true, y_pred):
    accuracy = np.round(np.sum(y_true == y_pred) / len(y_true),2)
    return accuracy*100
#%%

clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)

print("Accuracy:", acc)