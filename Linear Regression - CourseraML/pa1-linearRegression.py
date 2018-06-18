import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#loading data
data = np.genfromtxt("C:\Rohit\Github\Machine Learning Algorithms\Linear Regression - CourseraML\data\ex1data1.txt" , delimiter = ",")

#extracting x and labels from given data
x = np.array(data[:,[0]])
labels = np.array(data[:,[1]])
num_of_training_examples = len(y)

#adding one columns 
x = np.insert(x , 0 , values = 1 , axis = 1)
theta = np.zeros((2 , 1))


plt.plot(x , labels , "x" , color = 'r')
plt.xlabel("Population of city in 10,000s")
plt.ylabel("Profit in $10,000s")



def hypothesis(x , theta):
    return np.dot(x, theta)


def costfunction(x , y , theta):
    m = len(y)
    squared_error = ((hypothesis(x , theta) - y)**2)/ (2 * m)
    
    return (squared_error).sum()

    

costfunction(x , labels , theta)

def gradientDescent(x , y , learning_rate , num_iterations , theta):
    j = np.zeros((num_iterations  , 1))
    for i in range(num_iterations):
        h = hypothesis(x , theta) - y
        summation = np.dot(x.transpose() , h)
        theta -= (learning_rate / len(y)) * summation
        
        j[i] = computeCost(x , y ,theta)
        
    return theta , j 
	
