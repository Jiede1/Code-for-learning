import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'F:\coursera_homework\machine-learning-ex1\ex1\ex1data2.txt'  
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])  
data2.head()  

data2 = (data2 - data2.mean()) / data2.std()  
print data2.head()  

def computeCost(X, y, theta):  
	inner = np.power(((X * theta.T) - y), 2)
	return np.sum(inner) / (2 * len(X))

def gradientDescent(X, y, theta, alpha, iters):  
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost

data2.insert(0, 'Ones', 1)
print data2
# set X (training data) and y (target variable)
cols = data2.shape[1]  
X2 = data2.iloc[:,0:cols-1]  
y2 = data2.iloc[:,cols-1:cols]

# convert to matrices and initialize theta
X2 = np.matrix(X2.values)  
y2 = np.matrix(y2.values)  
theta2 = np.matrix(np.array([0,0,0])) 
 
print 'X2:',X2.shape
print 'y2:',y2.shape
print y2[0:10]

alpha=0.01
iters=1000
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

# get the cost (error) of the model
print computeCost(X2, y2, g2) 

yf=X2*g2.T
print yf[0:10]