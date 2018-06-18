import numpy as np
import matplotlib as plt 
import scipy.optimize , scipy.special
%matplotlib inline 

Path = #add path
data = np.genfromtxt(Path + "\ex2data2.txt" , delimiter = ",")
x = data[:, : 2]
labels = data[: , [2]]

#adding bias
x = np.insert(x , 0 , values = 1 , axis = 1)


x1 = x[:,[1]]
x2 = x[:,[2]]

X = mapFeatures(x1 , x2)
theta = np.zeros((X.shape[1] , 1))
lamb = 10

def sigmoid(x):
    return (1/(1 + np.exp(-x)))

def mapFeatures(x1 , x2):
    degree_of_polynomial = 6
    array = np.ones((len(x1) , 1))
    
    for i in range(1 , degree_of_polynomial+1):
        for j in range(0 , i+1):
            x1_term = x1 ** (i - j)
            x2_term = x2 ** (j)
            array = np.concatenate((array , (x1_term*x2_term)) , axis = 1)
    return array
            
    
def costFunction(theta , x, y):
    m = len(y)
    h = sigmoid(np.dot(x , theta))
    positive = -np.dot(y.transpose() , np.log(h))
    negative = -np.dot((1- y ).transpose() , np.log(1 - h))
    return ((positive + negative) / m).flatten()

def costFunctionRegularization(theta , x, y , lamb):
    m = len(y)
    cost = costFunction(theta , x , y)
    reg = (np.sum(theta **2)) * (lamb / (2 * m))
    return (cost + reg)

def gradientDescentRegularization(theta , x ,y, lamb):
    m = len(y)
    n = x.shape[1]
    theta = theta.reshape((n,1))
    h = sigmoid(np.dot(x , theta)) - y
    grad = np.dot(x.transpose() , h)  / m
    reg = (lamb / m) * np.sum(theta[1:])
    grad[1:] = grad[1:] + reg
    return (grad)
    
def minimize(theta , x , y , lamb):
    res = scipy.optimize.fmin_tnc(func=costFunctionRegularization, x0=theta, fprime=gradientDescentRegularization, args=(x, y, lamb))  
    return res
	
	

def predict(t, x): 
    m = X.shape[0]
    p = np.zeros((m,1))
    n = X.shape[1]
    t = t.reshape((n,1))
    h_theta = sigmoid(X.dot(t))
    for i in range(0, h_theta.shape[0]):
        if h_theta[i] > 0.5:
            p[i, 0] = 1
        else:
            p[i, 0] = 0
    return p


p = predict(a[0], X)
print ('Train Accuracy:', (labels[p == labels].size / float(labels.size)) * 100.0)