

# -*- coding: utf-8 -*-
""" Linear Regression using statmodels library
# Venki Uddameri, Lamar
"""
# Step 1: Load Libraries
import os 
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt

# Set working directory
path = '/media/vuddameri/4698-A2CF/LinearRegression/Code'
fname = 'ruraldensityspeed.csv'
os.chdir(path)

# Read data from csv file and extract variables
a = pd.read_csv(fname)
vobs = a['Speed']  
k = a['Density']

k = sm.add_constant(k) # Add a constant for intercept
model = sm.OLS(vobs,k) # Create the model
res = model.fit() # Fit the model to obtain predictions
print(res.summary()) # Write summary to console

vpred = res.fittedvalues.copy() # make a copy of predictions
err= vobs-vpred
plt.plot(vpred,err,'ro')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.grid()

# Harvey Collier Test for linearity
sm.stats.linear_harvey_collier(res)


#Breusch Pagan Test for homoskedasticity
BP = sm.stats.diagnostic.het_breuschpagan(err,k)

# tests of Normality
sm.qqplot(err,line='s')
plt.grid()

sm.stats.diagnostic.kstest_normal(err, dist='norm')

# Autocorrelation testing
sm.stats.diagnostic.acorr_breusch_godfrey(res) #need to provide regression model