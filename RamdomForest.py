'''随机森林需要调整的参数有：
（1）    决策树的个数
（2）    特征属性的个数
（3）    递归次数（即决策树的深度）'''

import numpy as np
from numpy import *
import random
from sklearn.cross_validation import train_test_split

#生成数据集。数据集包括标签，全包含在返回值的dataset上
def get_Datasets():
	from sklearn.datasets import make_classification
	dataSet,classLabels=make_classification(n_samples=200,n_features=100,n_classes=2)
	#print(dataSet.shape,classLabels.shape)
	return np.concatenate((dataSet,classLabels.reshape((-1,1))),axis=1)
	
#切分数据集，实现交叉验证。可以利用它来选择决策树个数。但本例没有实现其代码。
#原理如下：
#第一步，将训练集划分为大小相同的K份；
#第二步，我们选择其中的K-1分训练模型，将用余下的那一份计算模型的预测值，
#这一份通常被称为交叉验证集；第三步，我们对所有考虑使用的参数建立模型
#并做出预测，然后使用不同的K值重复这一过程。
#然后是关键，我们利用在不同的K下平均准确率最高所对应的决策树个数
#作为算法决策树个数
def splitDataSet(dataSet,n_folds):
	fold_size=len(dataSet)/n_folds
	data_split=[]
	begin=0
	end=fold_size
	for i in range(n_folds):
		data_split.append(dataSet[begin:end,:])
		begin=end
		end+=fold_size
	return data_split


#构建n个子集
def get_subsamples(dataSet,n):
	subDataSet=[]
	for i in range(n):
		index=[]
		for k in range(len(dataSet)):
			index.append(np.random.randint(len(dataSet)))
		subDataSet.append(dataSet[index,:])
	return subDataSet
	
#划分数据集
def binSplitDataSet(dataSet,feature,value):
	mat0=dataSet[np.nonzero(dataSet[:,feature]>value)[0],:]
	mat1=dataSet[np.nonzero(dataSet[:,feature]<value)[0],:]
	return mat0,mat1

#计算方差，回归时使用
def regErr(dataSet):
	return np.var(dataSet[:,-1])*shape(dataSet)[0]
#计算平均值，回归时使用
def regLeaf(dataSet):
	return np.mean(dataSet[:,-1])
def MostNumber(dataSet):  #返回多类
	#number=set(dataSet[:,-1])
	len0=len(np.nonzero(dataSet[:,-1]==0)[0])
	len1=len(np.nonzero(dataSet[:,-1]==1)[0])
	if len0>len1:
		return 0
	else:
		return 1
#计算基尼指数
def gini(dataSet):
	corr=0.0
	for i in set(dataSet[:,-1]):
		corr+=(len(np.nonzero(dataSet[:,-1]==i)[0])/len(dataSet))**2
	return 1-corr
	
#选取任意的m个特征，在这m个特征中，选取分割时的最优特征  
def select_best_feature(dataSet,m,alpha="huigui"):
	f=dataSet.shape[1]
	index=[]
	bestS=inf;bestfeature=0;bestValue=0;
	if alpha=="huigui":
		S=regErr(dataSet)
	else:
		S=gini(dataSet)
	for i in range(m):
		index.append(np.random.randint(f))
	for feature in index:
		for splitVal in set(dataSet[:,feature]):
			mat0,mat1=binSplitDataSet(dataSet,feature,splitVal)
			if alpha=="huigui":  newS=regErr(mat0)+regErr(mat1)
			else:
				newS=gini(mat0)+gini(mat1)
			if bestS>newS:
				bestfeature=feature
				bestValue=splitVal
				bestS=newS
	if (S-bestS)<0.001 and alpha=="huigui":    #如果误差不大就退出
		return None,regLeaf(dataSet)
	elif (S-bestS)<0.001:
		#print(S,bestS)
		return None,MostNumber(dataSet)
	#mat0,mat1=binSplitDataSet(dataSet,feature,splitVal)
	return bestfeature,bestValue

def createTree(dataSet,alpha="huigui",m=20,max_level=10):   #实现决策树，使用20个特征，深度为10
	bestfeature,bestValue=select_best_feature(dataSet,m,alpha=alpha)
	if bestfeature==None:
		return bestValue
	retTree={}
	max_level-=1
	if max_level<0:   #控制深度
		return regLeaf(dataSet)
	retTree['bestFeature']=bestfeature
	retTree['bestVal']=bestValue
	lSet,rSet=binSplitDataSet(dataSet,bestfeature,bestValue)
	retTree['right']=createTree(rSet,alpha,m,max_level)
	retTree['left']=createTree(lSet,alpha,m,max_level)
	#print('retTree:',retTree)
	return retTree

def RondomForest(dataSet,n,alpha="huigui"):   #树的个数
	#dataSet=get_Datasets()
	Trees=[]
	for i in range(n):
		X_train, X_test, y_train, y_test = train_test_split(dataSet[:,:-1], dataSet[:,-1], test_size=0.33, random_state=42)
		X_train=np.concatenate((X_train,y_train.reshape((-1,1))),axis=1)
		Trees.append(createTree(X_train,alpha=alpha))
	return Trees
	
#预测单个数据样本
def treeForecast(tree,data,alpha="huigui"):
	if alpha=="huigui":
		if not isinstance(tree,dict):
			return float(tree)
		if data[tree['bestFeature']]>tree['bestVal']:
			if type(tree['left'])=='float':
				return tree['left']
			else:
				return treeForecast(tree['left'],data,alpha)
		else:
			if type(tree['right'])=='float':
				return tree['right']
			else:
				return treeForecast(tree['right'],data,alpha)	
	else:
		if not isinstance(tree,dict):
			return int(tree)
		if data[tree['bestFeature']]>tree['bestVal']:
			if type(tree['left'])=='int':
				return tree['left']
			else:
				return treeForecast(tree['left'],data,alpha)
		else:
			if type(tree['right'])=='int':
				return tree['right']
			else:
				return treeForecast(tree['right'],data,alpha)	
#单棵树预测测试集				
def createForeCast(tree,dataSet,alpha="huigui"):
	m=len(dataSet)
	yhat=np.mat(zeros((m,1)))
	for i in range(m):
		yhat[i,0]=treeForecast(tree,dataSet[i,:],alpha)
	return yhat
	
#随机森林预测
def predictTree(Trees,dataSet,alpha="huigui"):
	m=len(dataSet)
	yhat=np.mat(zeros((m,1)))
	for tree in Trees:
		yhat+=createForeCast(tree,dataSet,alpha)
	if alpha=="huigui": yhat/=len(Trees)
	else:
		for i in range(len(yhat)):
			if yhat[i,0]>len(Trees)/2:
				yhat[i,0]=1
			else:
				yhat[i,0]=0
	return yhat

if __name__ == '__main__' :
	dataSet=get_Datasets()  #得到数据集和标签
	print(dataSet[:,-1].T)   #打印标签，与后面预测值对比
	RomdomTrees=RondomForest(dataSet,4,alpha="fenlei")   #4棵树，分类。
	print("---------------------RomdomTrees------------------------")
	#print(RomdomTrees[0])
	yhat=predictTree(RomdomTrees,dataSet,alpha="fenlei")
	print(yhat.T)
#get_Datasets()
	
	 
	 
	
	
	
	
	


