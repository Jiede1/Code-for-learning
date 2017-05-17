#coding:utf-8
import numpy as np
from math import log
import matplotlib.pyplot as plt
import operator

#计算给定数据集的熵
def calshang(dateSet):
	all=len(dataSet)
	labelCounts=[]
	label={}
	t=0.0
	for i in range(dataSet.shape[0]):
		num=dataSet[i,-1]
		labelCounts.append(num)
		if dataSet[i,-1] not in label.keys():
			label[dataSet[i,-1]]=0
		label[dataSet[i,-1]]+=1
	count=len(set(labelCounts))
	for key in label.keys():
		prob=float(label[key])/all
		t-=prob*log(prob,2)
	return t
#划分数据	
def splitdata(dataSet,axis,value):
	reData=[]
	
	for data in dataSet:
		if data[axis]==value:
			s1=[]
			s1.extend(data[:axis])
			s1.extend(data[axis+1:])
			reData.append(s1)
	return np.array(reData)
	
dataSet=np.array([[1,1,'Yes'],[1,1,'Yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']])
Labels=['no surfacing',' flipper']
print splitdata(dataSet,0,1)
print calshang(dataSet)

#选着最好的划分方式
def bestsplit(dataSet):
	m,n=dataSet.shape
	shang=calshang(dataSet)
	bestfeature=-1;bestinfogain=0.0
	for axis in range(n-1):
		p=dataSet[:,axis].tolist()
		t=set(p)
		for value in t:
			new=0.0
			reData=splitdata(dataSet,axis,value)
			prob=len(reData)/m
			new+=prob*calshang(reData)
		if shang-new>bestinfogain:
			bestinfogain=shang-new
			bestfeature=axis
	return bestfeature
	
#print bestsplit(dataSet)

def majority(dataSet):
	classCount={}
	classlist=[example[-1] for example in dataSet]
	for vote in classlist:
		if vote not in classCount.keys():
			classCount[vote]=0
		classCount[vote]+=1
	t=sorted(classCount.iteritems(),key=operator.itemgetter(0),reverse=True)
	return t[0][0]
#print majority(dataSet)
	

def creatTree(dataSet,Labels):
	print dataSet
	n=dataSet.shape[1]
	if len(set(dataSet[:,-1].tolist()))==1:return dataSet[0,-1]
	if n-1==0:return majority(dataSet)
	best=bestsplit(dataSet)
	bestLabel=Labels[best]
	tree={bestLabel:{}}
	del(Labels[best])
	featValues=[example[best] for example in dataSet]
	u=set(featValues)
	for i in u:
		sLabels=Labels[:]
		tree[bestLabel][i]=creatTree(splitdata(dataSet,best,i),sLabels)
	return tree

fr=open('E:\machine_learning_in_action_data\lab4\lenses.txt')
dataSet= np.array([inst.strip().split('\t') for inst in fr.readlines()])
Labels = ['age', 'prescript', 'astigmatic', 'tearRate']

tree=creatTree(dataSet,Labels)
print tree['age']['pre']['prescript']['hyper']
	
	 
	 
	 
	 
		
			
	
	