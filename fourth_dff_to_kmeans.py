#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:35:54 2021
load data from 'entireDF_DFF.csv' and use Column H which is abs dft output. 
This script will generate cluster
@author: jasonlee
"""



import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import os
import math
import ast
from scipy.fft import fft, ifft
from sklearn.cluster import KMeans
from ast import literal_eval

def toCsv(filename, df):
	cwd = os.getcwd()
	out_dir = os.path.join(cwd,filename+".csv")
	print(out_dir)
	df.to_csv(out_dir, index = None)


# Helper methods for centroids_finder
def distortionFinder(X):
    # X = X[:,[0,1]]
    distortions = []
    for i in range(1,30):
        km = KMeans(n_clusters=i, random_state=1).fit(X)
        distortions.append(km.inertia_)
    plt.plot(range(1, 30), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.tight_layout()
    plt.savefig('./dff_elbow.png', dpi=300)
    plt.show()
# Helper methods for centroids_finder
def inertiaFinder(X):
    # X = X[:,[0,1]]
    km = KMeans(n_clusters=20, random_state=1)
    distances= km.fit_transform(X)
    print(distances)
    variance = 0
    i=0
    retVal = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    retCount = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
# input df
def centroids_finder(arr, K):
    # print(arr)
    # distortionFinder(arr)
    distances, iVal, varianceVal, retVal , retCount = inertiaFinder(arr)
    # print(K)
    kmeans = KMeans(init='k-means++', n_clusters=K, random_state=1).fit(arr)
    labels = kmeans.labels_
    inertia = kmeans.inertia_
    # print(inertia)
    centroids = kmeans.cluster_centers_
    return centroids , labels , inertia, distances, iVal, varianceVal , retVal , retCount

# Not gon use
# def strToArr(df):
#     # df = df['abs_y_list']
#     arr = []
#     df['abs_y_list'] = df['abs_y_list'].apply(literal_eval)
#     for index, row in df.iterrows():
#         arr.append(row['abs_y_list'])
#         # print(type(row['[C,C#,D,E-,E,F,F#,G,G#,A,B-,B]']))
#     return arr


def main():
# 	dir_name = "haydnAnalysis"
	cwd = os.getcwd()
# 	directory = os.path.join(cwd,dir_name)
	entireDF = pd.read_csv("entireDF_DFF.csv")
	entireDF.head()
# 	replace [nan,] to 
	entireDF.at[0,'abs_y_list'] = "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
	dffCol = entireDF['abs_y_list']

	arr = []
	for row in dffCol:
		i = ast.literal_eval(row)
		arr.append(i)

	entireVP = arr
	type(entireVP)
	vectorPointE = np.asarray(entireVP)
	#check optimized number of centroid
	type(vectorPointE)
	
# 	This is abs dff result vec point. Start clustring from here
	vectorPointE
# 	find out the proper cluster# using elbow method
	distortionFinder(vectorPointE)
	
	
	
	
	
	centroidsVectorE, labelsArrayE, inertiaValueE, distancesE, iValE, varianceValE, retVal, retCount = centroids_finder(vectorPointE,20)
	
	for x in range(0,20):
		retVal[x] = retVal[x] / retCount[x]
		print(x)
	toCsv(centroidsVectorE,"centroidsVectorE")
	toCsv(labelsArrayE,"labelsArrayE")
	#makeCSV(inertiaValueE,"inertiaValueE")
	toCsv(distancesE,"distancesE")
    #makeCSV(iValE,"iValE")
    #makeCSV(varianceValE,"varianceValE")
	toCsv(retVal,"retVal")
	toCsv(retCount,"retCount")
	toCsv(entireDF, "entireDF")
    





if __name__ == '__main__':
    main()


