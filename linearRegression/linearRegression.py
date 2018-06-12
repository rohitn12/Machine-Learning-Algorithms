from numpy import *
 
def compute_error(b, m, points):
    # sum of sqaure errors 
    N = len(points)
    
    total_error = 0
    for i in range(0 , N):
        
        x = points[i , 0]
        y = points[i , 1]
        total_error += (y - (m*x + b))**2
        
    return total_error / float(N)

def step_gradient(current_b , current_m , points , learning_rate):
    #gradient descent computing local minima
    b_gradient = 0 
    m_gradient = 0 
    multiplication_factor = 2 / len(points)
    
    for i in range(0 , len(points)):
        x = points[i , 0]
        y = points[i , 1]
        # partial derivative wrt to b and m 
        b_gradient += -(multiplication_factor) * (y - (current_m *x) + current_b)
        m_gradient += -(multiplication_factor) * x * (y - (current_m *x) + current_b)
    #update new b and m learning rate defines how fast our model trains    
    new_m = current_m - (learning_rate * m_gradient)
    new_b = current_b - (learning_rate * b_gradient)
        
    return [ new_b , new_m]
        
    

def gradient_descent_runner(points  , initial_b , initial_m , learning_rate , num_of_iteration):
    # setting starting b and m values
    
    b = initial_b
    m = initial_m
     
    #getting the ideal b and m values 
    for i in range(num_of_iteration):
        b , m = step_gradient(b , m , array(points) , learning_rate)
    return [b,m]
def run():
    points = genfromtxt('C:\Rohit\Github\Machine Learning\Gradient descent\data.csv' , delimiter = ",")
    #hyperparamter learning rate defines how fast our model learns 
    learning_rate = .0001 
    # y = mx+ b (b is the intercept and m is the slope ) we are going to learn overtime
    initial_b = 0
    initial_m = 0
    num_of_iteration = 1000 #more number of iteration for larger dataset
    #getting ideal b and m values using gradient descent runner 
    print("initial error" , compute_error(initial_b , initial_m , points))
    [b, m] = gradient_descent_runner(points , initial_b , initial_m , learning_rate , num_of_iteration)
    print("b" , b)
    print("m" , m)
    print ("after gradient descent error is " ,  compute_error(b, m, points))

    #print("error after gradient descent" , compute_error(b , m , learning_rate))
if __name__ == '__main__':
    run()