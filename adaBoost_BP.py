#coding:utf-8
'''结合ADABOOST和BP的算法实现分类'''
import numpy as np 
from numpy import * 
import pandas as pd  
import matplotlib.pyplot as plt  

def loadDataSet(filename):
	dataMat=[]
	classLabels=[]
	fr=open(filename)
	for line in fr.readlines():
		lineArr=line.strip().split(',')
		dataMat.append([float(lineArr[0]),float(lineArr[1])])
		classLabels.append(int(lineArr[-1]))
	return mat(dataMat),mat(classLabels).T

X,classLabels=loadDataSet('F:\coursera_homework\machine-learning-ex2\ex2\ex2data2.txt')
y=classLabels

def sigmoid(z):  
    return 1 / (1 + np.exp(-z))
def result(h):
	result=[1 if t>=0.5 else 0 for t in h]
	return mat(result).T
def forward_propagate(X, theta1, theta2):  
    m = X.shape[0]

    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)
    #print h.T

    return a1, z2, a2, z3, h
def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):  
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # compute the cost
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)

    J = J / m
	#regularization
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2))) 

    return J
print X.shape
input_size = X.shape[1]
hidden_size = 4  
num_labels = 1 
learning_rate = 0.003

# randomly initialize a parameter array of the size of the full network's parameters
params = (random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25
#print 'p:',params.shape
m = X.shape[0]  

# unravel the parameter array into parameter matrices for each layer
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))  
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

def sigmoid_gradient(z):  
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):  
    ##### this section is identical to the cost function logic we already saw #####
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)
    #print shape(y)
    #print shape(h)
    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)

    J = J / m

    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))

    ##### end of cost function logic, below is the new part #####

    # perform backpropagation
    for t in range(m):
        a1t = a1[t,:]  # (1, 401)
        z2t = z2[t,:]  # (1, 25)
        a2t = a2[t,:]  # (1, 26)
        ht = h[t,:]  # (1, 10)
        yt = y[t,:]  # (1, 10)

        d3t = ht - yt  # (1, 10)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)

        delta1 = delta1 + (d2t[:,1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    # add the gradient regularization term
    delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * learning_rate) / m
    delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * learning_rate) / m

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2))) #(25*401+10*26=10285)
    #print 'grad:',grad.shape

    return J, grad
	
def BP(dataMat,classLabels,D):	
	from scipy.optimize import minimize
	bestStump={}
	# minimize the objective function
	fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y, learning_rate),  
					method='TNC', jac=True, options={'maxiter': 300})
	theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))  
	theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
	a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2) 
	#print h.T
	y_pred =result(h) 
	#print 'y_pred:',y_pred.T 	
	error = [0 if a == b else 1 for (a, b) in zip(y_pred, y)]  
	#print 'error:',len(error)
	erroRate = (sum(map(int, error)) / float(len(error)))
	print 'erroRate:',erroRate
	#print 'D,error:',D.shape,mat(error).T.shape
	minError=D.T*mat(error).T
	#print 'minError:',minError.shape
	#print minError
	bestStump['theta1']=theta1
	bestStump['theta2']=theta2
	bestStump['thresh']=0.5
	bestStump['erroRate']=erroRate
	
	return 	bestStump,minError,y_pred

#基于BP的ADABOOST训练过程
def adaBoostTrains(dataMat,classLabels,numIt=40):
	weakClassArr=[]#储存所有分类器数组
	m=shape(dataMat)[0]
	D=mat(ones((m,1))/m)#训练数据分配权重
	aggClassEst=mat(zeros((m,1)))#保存强分类器分类结果
	for i in range(numIt):
		bestStump,error,classEst=BP(dataMat,classLabels,D)
		#print "D:",D.T
		print 'error:',error
		alpha=float(0.5*log((1.0-error)/max(error,1e-16))) 
		
		bestStump['alpha']=alpha
		weakClassArr.append(bestStump)#保存分类器
		#print "classEst:",classEst.T
		p=multiply(-1*alpha*mat(classLabels),classEst)
		p=exp(p)
		#print 'p:',p.shape
		D=multiply(D,p)/D.sum()  #D(i)=D(i)*exp(-alpha*yi*classEst)/D.sum
		aggClassEst+=classEst*alpha
		#print 'aggClassEst: ',aggClassEst.T
		
		print aggClassEst.shape,classLabels.shape
		aggErrors=multiply(ones((m,1)),sign(aggClassEst)!=mat(classLabels))
		erroRate=aggErrors.sum()/m
		print "total rate:",erroRate,'\n'
		if erroRate==0.0: break
	return weakClassArr,aggClassEst #返回所有分类器
	
classify,aggClassEst=adaBoostTrains(X,y,40)