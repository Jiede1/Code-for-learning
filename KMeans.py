import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
#import seaborn as sb  
from scipy.io import loadmat  
#matplotlib inline

def find_closest_centroids(X, centroids):  
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)

    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i,:] - centroids[j,:]) ** 2)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j

    return idx
	
data = loadmat('F:\coursera_homework\machine-learning-ex7\ex7\ex7data2.mat')  
X = data['X']  
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

idx = find_closest_centroids(X, initial_centroids)  
print idx[0:3]  

def compute_centroids(X, idx, k):  
    m, n = X.shape
    centroids = np.zeros((k, n))

    for i in range(k):
        indices = np.where(idx == i)
        centroids[i,:] = (np.sum(X[indices,:], axis=1) / len(indices[0])).ravel()

    return centroids

compute_centroids(X, idx, 3)  

def run_k_means(X, initial_centroids, max_iters):  
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids

    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)

    return idx, centroids

idx, centroids = run_k_means(X, initial_centroids, 10)  

cluster1 = X[np.where(idx == 0)[0],:]  
cluster2 = X[np.where(idx == 1)[0],:]  
cluster3 = X[np.where(idx == 2)[0],:]

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(cluster1[:,0], cluster1[:,1], s=30, color='r', label='Cluster 1')  
ax.scatter(cluster2[:,0], cluster2[:,1], s=30, color='g', label='Cluster 2')  
ax.scatter(cluster3[:,0], cluster3[:,1], s=30, color='b', label='Cluster 3')  
ax.legend() 
plt.show() 

def init_centroids(X, k):  
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)

    for i in range(k):
        centroids[i,:] = X[idx[i],:]

    return centroids

init_centroids(X, 3) 

image_data=loadmat('F:\coursera_homework\machine-learning-ex7\ex7\ird_small.mat')  
print image_data
A = image_data['A']  
print A.shape  

# normalize value ranges
A = A / 255.

# reshape the array
X = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))

# randomly initialize the centroids
initial_centroids = init_centroids(X, 16)

# run the algorithm
idx, centroids = run_k_means(X, initial_centroids, 10)

# get the closest centroids one last time
idx = find_closest_centroids(X, centroids)

# map each pixel to the centroid value
X_recovered = centroids[idx.astype(int),:]

# reshape to the original dimensions
X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))
print 'X_recovered:',X_recovered.shape
plt.imshow(X_recovered)  
plt.show()

data = loadmat('F:\coursera_homework\machine-learning-ex7\ex7\ex7data1.mat')  
X = data['X']

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(X[:, 0], X[:, 1])  

def pca(X):  
    # normalize the features
    X = (X - X.mean()) / X.std()

    # compute the covariance matrix
    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]

    # perform SVD
    U, S, V = np.linalg.svd(cov)

    return U, S, V

U, S, V = pca(X)  
print U, S, V  
print 'U,X:',U.shape,X.shape
print type(X),type(U)
def project_data(X, U, k):  
    U_reduced = U[:,:k]
    return 	X*U_reduced				#np.dot(X, U_reduced)

Z = project_data(X, U, 1)  
print 'Z:',Z.shape  

def recover_data(Z, U, k):  
    U_reduced = U[:,:k]
    return np.dot(Z, U_reduced.T)

X_recovered = recover_data(Z, U, 1)  
#print 'X_recovered:',X_recovered  

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(X_recovered[:, 0], X_recovered[:, 1])
plt.show()  

faces = loadmat('F:\coursera_homework\machine-learning-ex7\ex7\ex7faces.mat')  
X = faces['X']  
print X.shape  

face = np.reshape(X[3,:], (32, 32))  
plt.imshow(face)
  

U, S, V = pca(X)
print 'U,S,V:',U.shape,S.shape,V.shape  
Z = project_data(X, U, 100)  

X_recovered = recover_data(Z, U, 100)  
face = np.reshape(X_recovered[3,:], (32, 32))  
plt.imshow(face) 
plt.show()
