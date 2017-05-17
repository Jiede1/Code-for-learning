#coding:utf-8
from numpy import *
import numpy as np
import matplotlib.pyplot as plt 
from scipy.io import loadmat  

data = loadmat('F:\coursera_homework\machine-learning-ex8\ex8\ex8data1.mat')  
X = data['X']  
print X.shape 
#print X 

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(X[:,0], X[:,1])  

def estimate_gaussian(X):  
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)

    return mu, sigma

mu, sigma = estimate_gaussian(X)  
print mu, sigma  

Xval = data['Xval']  
yval = data['yval']

#plt.bar(np.arange(1,3070,10),(Xval[:,1].tolist()),width = 0.35)
plt.hist(Xval[:,0].tolist())
#plt.stem(Xval[:,1].tolist())
from scipy import stats  
dist = stats.norm(mu[0], sigma[0]) #norm pdf两者合起来，计算X[0：50]对应的正态分布概率密度值 
print stats.norm.pdf(range(10,30),mu[0], sigma[0]) 
print dist.pdf(X[:,0])[0:50]  

p = np.zeros((X.shape[0], X.shape[1]))  
p[:,0] = stats.norm(mu[0], sigma[0]).pdf(X[:,0])  
p[:,1] = stats.norm(mu[1], sigma[1]).pdf(X[:,1])

print p.shape  

pval = np.zeros((Xval.shape[0], Xval.shape[1]))  
pval[:,0] = stats.norm(mu[0], sigma[0]).pdf(Xval[:,0])  
pval[:,1] = stats.norm(mu[1], sigma[1]).pdf(Xval[:,1])  

def select_threshold(pval, yval):  
    best_epsilon = 0
    best_f1 = 0
    f1 = 0

    step = (pval.max() - pval.min()) / 1000
    print pval.max()
    for epsilon in np.arange(pval.min(), pval.max(), step):
        preds = pval < epsilon #prediction result

        tp = np.sum(np.logical_and(preds == 1, yval == 1)).astype(float)
        fp = np.sum(np.logical_and(preds == 1, yval == 0)).astype(float)
        fn = np.sum(np.logical_and(preds == 0, yval == 1)).astype(float)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1

epsilon, f1 = select_threshold(pval, yval)  
print epsilon, f1  

outliers = np.where(p < epsilon)

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(X[:,0], X[:,1])  
ax.scatter(X[outliers[0],0], X[outliers[0],1], s=50, color='r', marker='o')  
plt.show() 