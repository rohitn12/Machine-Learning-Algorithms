from sklearn.svm import SVC
import scipy.optimize , scipy.io

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

Path = #add path




###Linear Kernel

data1 = scipy.io.loadmat(Path + "\ex6data1")
x1 = data1['X']
y1 = data1['y']

clf = SVC(C = 1 , kernel = "linear")
clf.fit(x1,y1.ravel())


###Gaussian Kernel

data2 = scipy.io.loadmat(Path + "\ex6data2")
x2 = data2['X']
y2 = data2['y']

clf = SVC(C = 50 , kernel = "rbf" , gamma = 6)
clf.fit(x2,y2.ravel())


###Polynomial Kernel

data3 = scipy.io.loadmat(Path + "\ex6data3")
x3 = data3['X']
y3 = data3['y']

clf = SVC(C = 1 , kernel = "poly" ,degree = 3 ,gamma = 10)
clf.fit(x3,y3.ravel())
