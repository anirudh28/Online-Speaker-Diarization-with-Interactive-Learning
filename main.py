# -*- coding: utf-8 -*-


import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import pathlib
import Linucb
import BerlinUCB
import BerlinUCB_GMM
import BerlinUCB_KMeans
import BerlinUCB_KNN

path=pathlib.Path(__file__).parent.absolute()
## Load the mat file corresponding to 15 speakers and epiReward as 0.1
mat = io.loadmat(str(path)+'/MAT_FILES/MFCC_15_0.1_10_0.mat')

## Use BerlinUCB Model on the loaded .mat file
reward_cumulative, accuracy_cumulative, t=BerlinUCB.BerlinUCB(mat)
t_array = np.arange(t)
data_mat=[reward_cumulative, accuracy_cumulative, t_array]
## Save the results in a csv file
np.savetxt(str(path)+"data_BerlinUCB_15_0.1_10_0.csv", data_mat , delimiter =",")

## For plotting use the GenerateResults.py file
