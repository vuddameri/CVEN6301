# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:32:05 2020
Multiple Linear Regression Example
@author: vuddameri 1/28/2020
"""

# load libraries
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Set working Directory
path = '/media/vuddameri/4698-A2CF/LinearRegression/Code'
fname = 'concretedata.csv'
os.chdir(path)
a = pd.read_csv(fname)
a.head(5) # explore first 5 rows

# Compute correlations and plot correlation matrix
corr = a.corr(method='kendall')
pd.DataFrame.to_csv(corr,'correl.csv') # Write to csv file
# source:https://seaborn.pydata.org/examples/many_pairwise_correlations.html
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})



# Perform exploratory scatterplot matrix
#pd.plotting.scatter_matrix(a,diagonal='kde')
#plt.tight_layout()
#plt.show()

# Compute Mutual Information Criteria
Y = a['StrengthMPa'] # strength is the Y variable
X = a.iloc[:,0:8] # slect first 8 columns (note the start and end)
MI = mutual_info_regression(X,Y)
MI = MI*100/np.max(MI)
cols = list(a.columns)[0:8]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(cols,MI)
plt.ylabel('Rel. Mutual Information')
plt.xticks(rotation='vertical')
plt.grid(True)
plt.show()


#Perform Regression using all variables
X = sm.add_constant(X)
mod = sm.OLS(Y,X)
res = mod.fit()
print(res.summary())
pred = res.fittedvalues.copy()
err = Y - pred
# Compute Variance Inflation Factors
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

# Run tests for LINE
plt.plot(pred,err,'ro')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.grid()

# Obs-Predicted plot
plt.plot(Y,pred,'bo')
plt.plot(Y, Y + 0, linestyle='solid')
plt.xlabel('Observed Strength MPa' )
plt.ylabel('Predicted Strength MPa' )
plt.grid()


#Breusch Pagan Test for homoskedasticity
BP = sm.stats.diagnostic.het_breuschpagan(err,X)

# tests of Normality
sm.qqplot(err,line='s')
plt.grid()

sm.stats.diagnostic.kstest_normal(err, dist='norm')

# Autocorrelation testing
sm.stats.diagnostic.acorr_breusch_godfrey(res) #need to provide regression model
