#coding:utf-8
import numpy as np
from numpy import *
import find_time_effective
from scipy.io import loadmat 
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
import time

start=time.clock()

#要验证是否Cluster03的思路是否正确：
#利用不同样本的FP1，FP2两条样本来证明，从有效记录点开始，
#对应的FP1/FP2通道，RGB对应段的相似度较大，否则对应段的相似度较小


#导入数据,数据采样率1000
dataset1=loadmat(u'E:\项目数据样本\chenminyi3.mat')  
dataset1=dataset1['EEG']['data'][0,0] #(64,1611720)

#dataset2=loadmat(u'E:\项目数据样本\liukeming01.mat')
#dataset2=dataset2['EEG']['data'][0,0]  #(64,3091120)

dataset3=loadmat(u'E:\项目数据样本\suyuxiao02.mat')  
dataset3=dataset3['EEG']['data'][0,0]  #(64,1647120)

dataset4=loadmat(u'E:\项目数据样本\yuli06.mat')   
dataset4=dataset4['EEG']['data'][0,0]  #(64,1614120)

D=find_time_effective.find_time_effective()
G=find_time_effective.split_dataset()
dataset=[dataset1,dataset3,dataset4]
Cd=[];FP1=[];FP2=[];X1=[];X2=[]
for i in range(3):   #提取出RGB数据矩阵（56*3，6000）为单位
	Cd[i]=D.time_effective(dataset[i])
	FP1[i],FP2[i],X1[i],X2[i]=G.splitnow(dataset[i],Cd[i])
	print(FP1[i].shape,FP2[i].shape)

def pearsSim(inA,inB):   #皮尔逊相似度,cov(x,y)/(sqrt(std(x))*sqrt(std(y)))
    if len(inA)<3:return 1.0
    return 0.5+0.5*np.corrcoef(inA,inB,rowvar=0)[0][1] 

#相似度比较
similarty1=0;similarty2=0;similarty3=0
resultF1={}
resultF2={}
resultX={}
for i in range(3):
	m,n=shape(FP1[i])
	for j in range(3):
		if i!=j:
			for k in range(m):
				similarty1+=pearsSim(FP1[i][k,:],FP1[j][k,:])
				similarty2+=pearsSim(X1[i][k,:],X2[j][k,:])
				similarty3+=pearsSim(FP2[i][k,:],FP2[j][k,:])
			similarty1/=m
			similarty2/=m
			similarty3/=m
			resultF1[str(i)+str(j)]=similarty1
			resultF2[str(i)+str(j)]=similarty3
			resultX[str(i)+str(j)]=similarty2

end=time.clock()	
print ("read:%fs" %(end-start))