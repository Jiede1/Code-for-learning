#coding:utf-8
from scipy.io import loadmat 
from sklearn.cluster import KMeans
from numpy import *
raw_data = loadmat('F:\BaiduNetdiskDownload\mat_files\chenminyi3.mat')  
print type(raw_data)
for key in raw_data:
	print key
print  type(raw_data['EEG'])
print  type(raw_data['EEG']['data'])
#print  raw_data['EEG']['data']
data=mat(raw_data['EEG']['data'][0,0])
print data.shape
data=data[1,:]
data=data[:,1000:]
m,n=data.shape          
print type(data[0,0:1200])
k=1
r=1
t=1
x=0   #用来找出RGB数据
dit=[[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
for iter in range(3944):
	X=[]	
	x=iter
	for i in range(3*7*8):
		if i%21.0!=0.0:
			line=array(data[0,x:x+1200])[0,:]
			X.append(line)
			x=x+1800
		else:
			x=x+10*200   #中间休息了10s
			line=array(data[0,x:x+1200])[0,:]
			X.append(line)
			x=x+1800
	X=mat(X)
	y_pred = KMeans(n_clusters=3).fit_predict(X)
	if k==1:
		print X.shape
		k=0
	if r==1:
		print y_pred.T
		print type(y_pred[0:2])
		print y_pred.shape
		print y_pred[0:3].tolist() 
		print (y_pred[3:6]-array([0,1,2]))
		r=0
	'''if y_pred[0:3].tolist() in dit:
		print True
	else:
		print False
		print y_pred[0:3].tolist()'''
	if y_pred[0:3].tolist() in dit:
		print y_pred[0:3]
		print iter
		print y_pred
		'''if y_pred[3:6].tolist() in dit:
			print y_pred[3:6]
			if y_pred[6:9].tolist() in dit:
				print y_pred'''
		
		

	







