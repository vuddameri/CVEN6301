#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 15:49:44 2023
A script to work with SSURGO datasets
Example of extracting Soil Texture
@author: vuddameri
"""
# Import libraries
import geopandas as gpd
import fiona
import pandas as pd
import os

# Set working directory
# Change to suit your situation
dir = '/media/vuddameri/EXTHD/MachineLearning'
os.chdir(dir)

# Provide the full path of gSSURGO file
# Change to suit your situation
fname = '/media/vuddameri/EXTHD/MachineLearning/gSSURGO_TX.gdb'
layers = fiona.listlayers(fname) # use fiona to list layers

# Extract those layers that are needed for soil texture
# look at the SSURGO Data model - 
# https://www.nrcs.usda.gov/sites/default/files/2022-08/SSURGO-Data-Model-Diagram-Part-1_0_0.pdf
# Caution - These are slow operations - Please be patient
texture = gpd.read_file(fname,layer = 'chtexture')
texturegrp = gpd.read_file(fname,layer = 'chtexturegrp')
horizon = gpd.read_file(fname,layer = 'chorizon')
component = gpd.read_file(fname,layer='component')
mapunit = gpd.read_file(fname,layer='mapunit')
soilpoly = gpd.read_file(fname,layer='MUPOLYGON')

#plot the Soil Polygon

plotx = soilpoly.plot()
plotx.set_xlabel('Easting (m)')
plotx.set_ylabel('Northning (m)')

# Perform Joins to get Soil Texture with the map
texturegrp1 = pd.merge(texturegrp,texture[['texcl','chtgkey']],
                       on='chtgkey',how='left')
horizon1 = pd.merge(horizon,texturegrp1[['texcl','chkey']],on='chkey',
                    how='left')
component1 = pd.merge(component,horizon1[['texcl','cokey']],on='cokey',
                    how='left')
mapunit1 = pd.merge(mapunit,component1[['texcl','mukey']],on='mukey',
                    how='left')

mapunit1.to_csv('soiltex.csv')  # save mapunit1 as csv


# extract the crs and geometry
crsx = soilpoly.crs
geomx = soilpoly.geometry

soilpoly1 = pd.merge(soilpoly,mapunit1,on='mukey',how='inner')
soilpoly1.set_geometry(geomx,inplace=True,crs = crsx)
soilpoly1.plot('texcl')



