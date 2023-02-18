# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:18:02 2020
Logisitic Regression Example
@author: vuddameri 02/15/2023
"""

# Load Libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from itertools import chain
from matplotlib import pyplot as plt


# CHange working directory
dir = '/media/vuddameri/EXTHD/MachineLearning/LogisticRegression/Code'
os.chdir(dir)

# Read the dataset
a = pd.read_csv('TXculvertdata.csv') # read our dataset
features = ['SVCYR','ADT','Reconst','PTRUCK'] # INPUT DATA FEATURES
X = a[features] # DATAFRAME OF INPUT FEATURES
SVCYR2 = a['SVCYR']  # Add SVCYR square to the dataset
SVCYR2 = np.power(SVCYR2,2)
X.insert(1,"SVCYR2",SVCYR2)  # Insert the data frame
Y = a['Culvert_Damage'] # ADD IT TO THE INPUT FEATURE DATAFRAME

# Split into training and testing data
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.30,
                                               random_state=10)

# instantiate the model object (using the default parameters)
logreg = LogisticRegression(C=10**9) # Large C means no regularization

# fit the model with data
logreg.fit(X_train,y_train)

# Make Predictions
y_pred=logreg.predict(X_test) # Make Predictions
yprob = logreg.predict_proba(X_test) #test output probabilities
zz = pd.DataFrame(yprob)
zz.to_csv('probs.csv')

# Get the parameters
logreg.get_params()
logreg.coef_
logreg.intercept_

# Write the data to a file
keys = list(X.columns)
keys.append('Intercept')

vals = logreg.coef_.tolist()
vals = list(chain.from_iterable(vals))
intcept = float(logreg.intercept_)
vals.append(intcept)

par_dict = dict(zip(keys,vals))
with open('pars.txt','w') as data: 
      data.write(str(par_dict))

# Create a confusion Matrix and plot it
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix # y_test is going be rows (obs), y_pred (predicted) are cols
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)
disp.plot()
plt.show()

# Evaluate usng accuracy, precision, recall
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) # overall accuracy
print("Precision:",metrics.precision_score(y_test, y_pred)) # predicting 0 (Sat)
print("Recall:",metrics.recall_score(y_test, y_pred)) # predicting 1 (unsat)

# ROC Curve
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(round(auc,4)))
plt.legend(loc=4)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.grid()
plt.show()







