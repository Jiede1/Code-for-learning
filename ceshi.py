from numpy import *
test=mat([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
print 'test0',test[0]
mat0=test[nonzero(test[:,1]>0.5)[0],:][0]
print mat0
print test[:,1]>0.5
print nonzero(test[:,1]>0.5)
print '123',test[nonzero(test[:,1]>0.5)[0],:]
print test[:,-1].T.tolist()[0]
test1=mat([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
print test1[:,-1]-1
aa=array([[1,2]])
print aa.tolist()