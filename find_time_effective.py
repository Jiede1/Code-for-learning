#coding:utf-8
import numpy as np
from numpy import *
from sklearn.cluster import Birch


def pearsSim(inA,inB):   #皮尔逊相似度,cov(x,y)/(sqrt(std(x))*sqrt(std(y)))
    if len(inA)<3:return 1.0
    return 0.5+0.5*np.corrcoef(inA,inB,rowvar=0)[0][1] 

class find_time_effective(object):
	def __init__(self):
		self.shiyantime=159200
		
	def time_effective(self,dataset):
		m,n=shape(dataset)
		FP1=dataset[0]  #FP1电极
		FP2=dataset[2]		

		Birch_algorithm=Birch(threshold=0.5, branching_factor=56, n_clusters=3, compute_labels=True, copy=True)

		r3=[]  #最为关键，预测的两个labels的相似度
		
		for i in range(0,n-self.shiyantime+1,5):   #我有点担心复杂度的问题，运行太长时间
			#用于储存分割后的数据
			d1=[]
			d2=[]
			
			#用于储存相似度
			r1=[]   #labels与FP1预测的labels的相似度
			r2=[]
			
			start=i;end=i+6000
			for j in range(56*3): 
				d1.extend(FP1[start:end].tolist())  #tolist() flatten array to list,size not changed
				#print j,len(FP1[start:end].tolist())
				d2.extend(FP2[start:end].tolist())
				if j%21.0!=0:
					start=end+3000
					end=start+6000  #每张图片维持6秒
				else:   #凑够7次，休息10秒
					start=end+10000
					end=start+6000
			#print len(d1)
			d1=np.mat(d1).reshape((56*3,-1))   #用于聚类的数据，与labels对应
			d2=np.mat(d2).reshape((56*3,-1))
			#print d1.shape,d2.shape,i
			
			#开始执行聚类，基于实习时的经验，使用Birch聚类效果可能最好(Large dataset, outlier removal, data reduction)，3类
			
			Birch_algorithm.fit(d1)
			labels_predict1=Birch_algorithm.predict(d1)
			
			Birch_algorithm.fit(d2)
			labels_predict2=Birch_algorithm.predict(d2)
			
			#开始执行相似度聚类，查阅文献最合适的是皮尔逊系数，尽管表现不是最理想的。但能衡量相似度
			#r1.append(pearsSim(labels,labels_predict1))
			
			#r2.append(pearsSim(labels,labels_predict2))
			
			r3.append(pearsSim(labels_predict1,labels_predict2))
			#print pearsSim(labels_predict1,labels_predict2)
			#print r3

			#Cd1=r1.index(max(r1))
			#Cd2=r2.index(max(r2))
		Cd3=r3.index(max(r3))
		print(Cd3)
		return Cd3

class split_dataset(object):
	def __init__(self):
		self.Cd=0
	def splitnow(self,dataset,Cd):
		self.Cd=Cd
		d1=[];d2=[]
		X1=[];X2=[]
		
		FP1=dataset1[0]  #FP1电极
		FP2=dataset1[2]	
		
		start=Cd;end=Cd+6000
		for j in range(56*3): 
			d1.extend(FP1[start:end].tolist())
			X1.extend(FP1[end:end+3000].tolist())
			
			#print j,len(FP1[start:end].tolist())
			
			d2.extend(FP2[start:end].tolist())
			X2.extend(FP2[end:end+3000].tolist())
			
			if j%21.0!=0:
				start=end+3000
				end=start+6000  #每张图片维持6秒
			else:   #凑够7次，休息10秒
				start=end+10000+3000  #需要加上3s的休息步骤
				end=start+6000
		#print len(d1)
		d1=np.mat(d1).reshape((56*3,-1))   #用于聚类的数据，与labels对应
		d2=np.mat(d2).reshape((56*3,-1))
		#print d1.shape,d2.shape,i
		X1=np.mat(X1).reshape((56*3,-1))
		X2=np.mat(X2).reshape((56*3,-1))
		
		return d1,d2,X1,X2   #返回FP1，FP2，以及3秒的休息时间	
		
