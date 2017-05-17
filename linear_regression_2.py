import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

def func(x,p):
	k,b=p
	y=k*x+b
	return y

def residual(p,x,y):
	S=y-func(x,p)
	return S

def loadData(filename):
	fr=open(filename)
	dataSet=[]
	classLabels=[]
	for line in fr.readlines():
		line=line.strip().split(',')
		dataSet.append(float(line[0]))
		classLabels.append(float(line[1]))
	return np.array(dataSet),np.array(classLabels)

p0=np.random.random((2,1))
X,classLabels=loadData('E:\coursera_machine_learning\machine-learning-ex1\ex1\ex1data1.txt')
result=leastsq(residual,p0,args=(X,classLabels))
print result[0][0]




plt.scatter(X,classLabels,color='red',label='Sample point')
x=np.linspace(min(X)-1,max(X)+1,1000)
y=result[0][0]*x+result[0][1]
plt.plot(x,y,color='yellow',label='Fitting Line',linewidth=5)
plt.legend()
plt.xlabel('X')
plt.ylabel('classLabels')
plt.show()



