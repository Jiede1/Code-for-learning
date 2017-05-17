import numpy as np

x=np.random.random((10,120))
x_mean=x-x.mean(0)
zx=np.cov(x,rowvar=0) #(120,120)
print(zx.shape)
U,S,V=np.linalg.svd(zx)
print(U.shape,S.shape,V.shape) #te zheng zhi
r=np.mat(x_mean)*np.mat(U[:,0:5])
print r.shape

eigVal,eigVet=np.linalg.eig(zx)
print eigVet.shape
rr=np.mat(x_mean)*np.mat(eigVet[:,0:5].real)
print rr.shape

print rr==r
print r,'\n'
print rr

print eigVet[:,:5]
print eigVet[:5,:5].flatten()
print U[:5,:5].flatten()
