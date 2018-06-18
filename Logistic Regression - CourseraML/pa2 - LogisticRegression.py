import numpy as np
import matplotlib as plt
import scipy.optimize , scipy.special
%matplotlib inline

Path = #add path to data
data = np.genfromtxt(Path + "\ex2data1.txt" , delimiter = ",")
m = np.shape(data)[0]
n = np.shape(data)[1] -1


x = data[:, : n]
labels = data[:,[n]]
#adding bias 
x = np.insert(x , 0 , values = 1 , axis = 1)

init_theta = np.zeros((n+1 ,1))

def sigmoid(x):
    return (1/ (1 + np.exp(-x)))

def cost_function(theta , x , y ):
    m = len(y)
    hypothesis = sigmoid(np.dot(x , theta))
    positive_term = -np.dot(y.transpose() , np.log(hypothesis))
    negative_term = -np.dot((1-y).transpose() , np.log(1 - hypothesis))
    
    return ((positive_term + negative_term) / m).flatten()

def gradientDescent(theta , x , y ):
    hypothesis = sigmoid(np.dot(x , theta)) - y
    return  (np.dot(x.transpose() , hypothesis)) / np.shape(x)[0]


def optimizeTheta(theta , x , y):
    result = scipy.optimize.fmin( cost_function, x0=theta, args=(x, y), maxiter=500, full_output=True )   
    return result[0] , result[1]
    
def predict(x , theta):
    prediction_prob = sigmoid(np.dot(x , theta))
    if prediction_prob > 0.5:
        return 1
    else:
        return 0
