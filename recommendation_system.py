#coding:utf-8
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
#import seaborn as sb  
from scipy.io import loadmat  
#matplotlib inline

data = loadmat('F:\coursera_homework\machine-learning-ex8\ex8\ex8_movies.mat')  
print data
Y = data['Y']  
R = data['R']  
print Y.shape, R.shape  

def cost(params, Y, R, num_features, learning_rate):  
    Y = np.matrix(Y)  # (1682, 943)
    R = np.matrix(R)  # (1682, 943)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]

    # reshape the parameter array into parameter matrices
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))  # (1682, 10)
    Theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))  # (943, 10)

    # initializations
    J = 0
    X_grad = np.zeros(X.shape)  # (1682, 10)
    Theta_grad = np.zeros(Theta.shape)  # (943, 10)

    # compute the cost
    error = np.multiply((X * Theta.T) - Y, R)  # (1682, 943)
    squared_error = np.power(error, 2)  # (1682, 943)
    J = (1. / 2) * np.sum(squared_error)

    # add the cost regularization
    J = J + ((learning_rate / 2) * np.sum(np.power(Theta, 2)))
    J = J + ((learning_rate / 2) * np.sum(np.power(X, 2)))

    # calculate the gradients with regularization
    X_grad = (error * Theta) + (learning_rate * X)
    Theta_grad = (error.T * X) + (learning_rate * Theta)

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))

    return J, grad

movie_idx = {}  
f = open('F:\coursera_homework\machine-learning-ex8\ex8\movie_ids.txt')  
for line in f:  
    tokens = line.split(' ')
    tokens[-1] = tokens[-1][:-1]
    movie_idx[int(tokens[0]) - 1] = ' '.join(tokens[1:])

ratings = np.zeros((1682, 1))

ratings[0] = 4  
ratings[6] = 3  
ratings[11] = 5  
ratings[53] = 4  
ratings[63] = 5  
ratings[65] = 3  
ratings[68] = 5  
ratings[97] = 2  
ratings[182] = 4  
ratings[225] = 5  
ratings[354] = 5

print('Rated {0} with {1} stars.'.format(movie_idx[0], str(int(ratings[0]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[6], str(int(ratings[6]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[11], str(int(ratings[11]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[53], str(int(ratings[53]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[63], str(int(ratings[63]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[65], str(int(ratings[65]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[68], str(int(ratings[68]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[97], str(int(ratings[97]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[182], str(int(ratings[182]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[225], str(int(ratings[225]))))  
print('Rated {0} with {1} stars.'.format(movie_idx[354], str(int(ratings[354]))))  

R = data['R']  
Y = data['Y']

Y = np.append(Y, ratings, axis=1)   #Y增加1列
R = np.append(R, ratings != 0, axis=1) #R增加1列 

print Y.shape,R.shape
print Y

from scipy.optimize import minimize

movies = Y.shape[0]  
users = Y.shape[1]  
features = 10  
learning_rate = 10.

X = np.random.random(size=(movies, features))  
Theta = np.random.random(size=(users, features))  
params = np.concatenate((np.ravel(X), np.ravel(Theta)))

Ymean = np.zeros((movies, 1))  
Ynorm = np.zeros((movies, users))

for i in range(movies):  
    idx = np.where(R[i,:] == 1)[0]
    Ymean[i] = Y[i,idx].mean()
    Ynorm[i,idx] = Y[i,idx] - Ymean[i]

fmin = minimize(fun=cost, x0=params, args=(Ynorm, R, features, learning_rate),method='CG', jac=True, options={'maxiter': 250})

X = np.matrix(np.reshape(fmin.x[:movies * features], (movies, features)))  
Theta = np.matrix(np.reshape(fmin.x[movies * features:], (users, features)))

print X.shape, Theta.shape  

predictions = X * Theta.T  
#print predictions+Ymean
my_preds = predictions[:, -1] + Ymean 
print my_preds 
sorted_preds = np.sort(my_preds, axis=0)[::-1]  
#print sorted_preds[:10]  
