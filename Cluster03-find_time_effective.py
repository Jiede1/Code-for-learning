#coding:utf-8
'''取chenminyi3的数据进行聚类，目标是找出时间有效记录点
代码思路：
横向聚类。将看到RGB波段的数据分段R，G，B出来，
将单个人的FP1的单个段视为一个样本点，
对单个人所有的数据进行聚类。
然后对FP2也进行同样的操作。
对比他们的聚类结果。在有效记录时间点的时候相似度理论上最大。'''

from scipy.io import loadmat 
from sklearn.cluster import KMeans
from numpy import *
import numpy as np
from sklearn.cluster import Birch

def pearsSim(inA,inB):   #皮尔逊相似度,cov(x,y)/(sqrt(std(x))*sqrt(std(y)))
    if len(inA)<3:return 1.0
    return 0.5+0.5*np.corrcoef(inA,inB)[0][1] 

dataset1=loadmat(u'E:\项目数据样本\chenminyi3.mat')  #10
dataset1=dataset1['EEG']['data'][0,0] #(64,1611720)
print (dataset1.shape)
FP1=dataset1[0]  #FP1电极
FP2=dataset1[2]

#先验标签
labels=np.array([0,1,2]*56)

Birch=Birch(threshold=0.5, branching_factor=56, n_clusters=3, compute_labels=True, copy=True)

r3=[]  #最为关键，预测的两个labels的相似度

#for i in range(0,1611720-1592000,10):   #我有点担心复杂度的问题，运行太长时间
for i in [19720]:
	#用于储存分割后的数据
	d1=[]
	d2=[]
	
	#用于储存相似度
	r1=[]   #labels与FP1预测的labels的相似度
	r2=[]
	
	
	start=i;end=i+6000
	for j in range(56*3): 
		d1.extend(FP1[start:end].tolist())
		#print j,len(FP1[start:end].tolist())
		d2.extend(FP2[start:end].tolist())
		if j%21!=0 or j==0:  #0%21.0=0
			start=end+3000
			end=start+6000  #每张图片维持6秒
		else:   #凑够7次，休息10秒
			start=end+10000+3000
			end=start+6000
		if j==56*3-1:
			print(start,end,start+10000)
			print(len(FP1[start:end].tolist()))
	#print len(d1)
	d1=np.mat(d1).reshape((56*3,6000))   #用于聚类的数据，与labels对应

	d2=np.mat(d2).reshape((56*3,6000))
	print (d1.shape,d2.shape,i)
	
	#开始执行聚类，基于实习时的经验，使用Birch聚类效果可能最好(Large dataset, outlier removal, data reduction)，3类
	
	Birch.fit(d1)
	labels_predict1=Birch.predict(d1)
	
	Birch.fit(d2)
	labels_predict2=Birch.predict(d2)
	
	#开始执行相似度聚类，查阅文献最合适的是皮尔逊系数，尽管表现不是最理想的。但能衡量相似度
	#r1.append(pearsSim(labels,labels_predict1))
	
	#r2.append(pearsSim(labels,labels_predict2))
	
	r3.append(pearsSim(labels_predict1,labels_predict2))
	#print pearsSim(labels_predict1,labels_predict2)
	#print r3

#Cd1=r1.index(max(r1))
#Cd2=r2.index(max(r2))
Cd3=r3.index(max(r3))
print ('the maximum index:',Cd3)  #测到数据是1730
	
	
	



		