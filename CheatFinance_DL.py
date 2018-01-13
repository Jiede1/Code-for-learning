#coding:utf-8
from __future__ import division  
import numpy as np   
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.cross_validation import train_test_split 
import seaborn as sns
#%matplotlib inline
file=pd.read_csv(r"E:\important_dataset\creditcard.csv")

#Time,V1...V28,Amount,Class  (284807,31)
#print(file.head())   #(5,31)
print("打印出数据集一些信息：")
print(file.describe())
print(file.head(1))

####数据预处理####
print("检查数据是否有缺失值")
print(file.isnull().sum())
file=file.drop(file.columns[0],axis=1)  #去掉Time列
#print(file.shape)  (284807,30)
file['Amount']=StandardScaler().fit_transform(file['Amount'].values.reshape(-1, 1))  #归一化Amount列

###数据可视化###
count_classes = pd.value_counts(file['Class'], sort = True).sort_index()
count_classes.index=['normal','cheat']  #更新index
print("两类各占比")
print(count_classes)
count_classes.plot(kind='bar')   #数据类别非常不均衡

corr = file.corr()#特征的相关系数矩阵 
f, ax = plt.subplots(figsize=(10, 10)) 
cmap = sns.diverging_palette(220, 10, as_cmap=True) 
sns.heatmap(corr, cmap=cmap, vmax=1.0, square=True, xticklabels=2, yticklabels=2, linewidths=.3, cbar_kws={"shrink": .5}, ax=ax) 
plt.show()

'''
def data_prepration(x): 
	x_features= x.ix[:,x.columns != "Class"]
	x_labels=x.ix[:,x.columns=="Class"]           
	x_features_train,x_features_test,x_labels_train,x_labels_test = train_test_split(x_features,x_labels,test_size=0.3)
	print("length of training data")
	print(len(x_features_train))
	print("length of test data")
	print(len(x_features_test))
	return(x_features_train,x_features_test,x_labels_train,x_labels_test)
data_train_X,data_test_X,data_train_y,data_test_y=data_prepration(file)
print (pd.value_counts(data_test_y['Class']))
#调用smote
os = SMOTE(random_state=0) 
print(type(data_train_X.values),data_train_X.values.shape)  #numpy,(199364,29)
os_data_X,os_data_y=os.fit_sample(data_train_X.values,data_train_y.values.ravel())
columns = data_train_X.columns
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
print (len(os_data_X))
os_data_y= pd.DataFrame(data=os_data_y,columns=["Class"])
# 现在检查下抽样后的数据
print("length of oversampled data is ",len(os_data_X))
print("Number of normal transcation",len(os_data_y[os_data_y["Class"]==0]))
print("Number of fraud transcation",len(os_data_y[os_data_y["Class"]==1]))
print("Proportion of Normal data in oversampled data is ",len(os_data_y[os_data_y["Class"]==0])/len(os_data_X))
print("Proportion of fraud data in oversampled data is ",len(os_data_y[os_data_y["Class"]==1])/len(os_data_X))
'''

#newtraindata=pd.concat([os_data_X,os_data_y],axis=1)
#newtestdata=pd.concat([data_test_X,data_test_y],axis=1)
#newtraindata.to_csv(r'C:\Users\Administrator\Desktop\train.csv',sep=',')
#newtestdata.to_csv(r'C:\Users\Administrator\Desktop\test.csv',sep=',')
