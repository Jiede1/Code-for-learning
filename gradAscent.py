#coding:UTF-8
#-*- coding: cp936 -*-
from numpy import *
import matplotlib.pyplot as plt

def sigmoid(inX): 
	return 1.0/(1+exp(-inX))
	
def loadDataSet(filename):
 
	dataMat = []; labelMat = []

# 打开文件
	fr = open(filename)

# 遍历数据文件每一行
	for line in fr.readlines():

    # 根据制表符切分每一行的数据
		currLine = line.strip().split('\t')
		lineArr =[]

    # 将21项数据都去进lineArr列表
		for i in range(21):
			lineArr.append(float(currLine[i]))

    # 添加到矩阵和类别标签列表
		dataMat.append(lineArr)

    # 第21列为类别
		labelMat.append(float(currLine[21]))
	return array(dataMat), labelMat

def stocGradAscent1(dataMatrix, classLabels, numIter=250):

# 首先获取训练数据的数量及特征数量
	m,n = shape(dataMatrix)
#定义回归·矩阵
	hmatrix=[];x=[]

# 初始化回归系数向量，向量长度为特征数量
	weights = ones(n) 

# 训练循环numIter次
	for j in range(numIter):
		dataIndex = range(m)

    # 遍历所有数据
		for i in range(m):
        # 每次迭代时调整alpha值
        # 每次调整后alpha值都会减小
			alpha = 4/(1.0+j+i)+0.0001
        # 获得随机数
        # 随机选取样本并对回归系数进行更新
			randIndex = int(random.uniform(0,len(dataIndex)))
        # 调用Sigmoid公式计算样本预测值
			h = sigmoid(sum(dataMatrix[randIndex]*weights))
        # 对比真实的分类和预测值
			error = classLabels[randIndex] - h
        # 调整回归系数
			weights = weights + alpha * error * dataMatrix[randIndex]
			
			hmatrix.append(weights)
			x.append(j*numIter+i)
			
        # 删除已用过的样本数据
			del(dataIndex[randIndex])
#画出回归系数与迭代次数的关系
	print hmatrix[10]
	print x[0:20]
	y1=mat(hmatrix)
	y=y1[:,2]
	fig=plt.figure()
	ax=fig.add_subplot(311)
	ax.plot(x,y)
	plt.show()
	return weights

def classifyVector(inX, weights):

# 计算Sigmoid公式
	prob = sigmoid(sum(inX*weights))
# 阈值为0.5，大于0.5则判断为1.0
	if prob > 0.5: return 1.0
	else: return 0.0

def colicTest():

# 读取训练数据
	trainingSet, trainingLabels = loadDataSet(u'E:\机器学习数据\疝气病数据\lab7\horseColicTraining.txt')

# 使用改进随机梯度上升（下降）法训练1000次获得最优的回归系数
	trainWeights = stocGradAscent1(trainingSet, trainingLabels, 1000)

# 初始化错误率及测试条目数量
	errorCount = 0; numTestVec = 0.0

# 打开测试数据
	frTest = open(u'E:\机器学习数据\疝气病数据\lab7\horseColicTest.txt')

# 遍历读取测试数据
	for line in frTest.readlines():
    # 增加条目计数
		numTestVec += 1.0

    # 根据制表符切分每一行的数据
		currLine = line.strip().split('\t')
		lineArr =[]

    # 将21项数据都去进lineArr列表
		for i in range(21):
			lineArr.append(float(currLine[i]))

    # 使用分类器进行预测
    # 对比分类预测结果及真实结果并记录错误率
		if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
			errorCount += 1

# 计算并输出错误率
	errorRate = (float(errorCount)/numTestVec)
	print "the error rate of this test is: %f" % errorRate
	return errorRate

def multiTest(): 
	numTests = 10; errorSum=0.0 
	for k in range(numTests): 
		errorSum += colicTest() 
	print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))
	
multiTest()