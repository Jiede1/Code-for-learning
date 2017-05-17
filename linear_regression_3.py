import numpy as np
import matplotlib.pyplot as plt

def loadData(filename):
	fr=open(filename)
	dataSet=[]
	classLabels=[]
	for line in fr.readlines():
		line=line.strip().split(',')
		dataSet.append(float(line[0]))
		classLabels.append(float(line[1]))
	return np.array(dataSet),np.array(classLabels)

def func(x,p):
	k,b=p
	return k*x+b
	
def Cost(dataSet,y,p):
	m=np.shape(dataSet)[0]
	J=np.sum(np.power(func(dataSet,p)-y,2))/(2*m)
	return J
	
def gradient(dataSet,y,p):
	m=np.shape(dataSet)[0]
	t=np.zeros((2,1))
	t[1]=np.sum(func(dataSet,p)-y)/m
	t[0]=np.sum(np.multiply(func(dataSet,p)-y,dataSet))/m
	#print 't:',t
	return t
	


def feature_norm(dataSet):
	meanv=np.mean(dataSet)
	stdv=np.std(dataSet)
	dataSet=(dataSet-meanv)/stdv
	return dataSet
	
	
def gradient_algorithm(x,classLabels,p,maxiter=1500,alpha=0.01):
	#x=feature_norm(x)
	#classLabels=feature_norm(classLabels)
	for i in range(maxiter):
		p=p-alpha*gradient(x,classLabels,p)
		#print p
		cost.append(Cost(X,classLabels,p))
	return p,cost
	
X,classLabels=loadData('E:\coursera_machine_learning\machine-learning-ex1\ex1\ex1data1.txt')
p=np.random.random((2,1))
print np.sum(np.power(p[1]*X+p[0]-classLabels,2))
cost=[]
p,cost=gradient_algorithm(X,classLabels,p)
print len(cost)
print cost[-1]
print p

plt.subplot(211)
plt.scatter(X,classLabels,color='red',label='Sample point')
x=np.linspace(min(X)-1,max(X)+1,1000)
y=p[0]*x+p[1]
plt.plot(x,y,color='yellow',label='Fitting Line')
plt.legend()
plt.xlabel('X')
plt.ylabel('classLabels')
plt.subplot(212)
times=np.linspace(1,len(cost),len(cost))
plt.plot(times,cost,label='cost')
plt.show()

		
