#coding:utf-8
'''结合ADABOOST和DECISION STUMP的算法实现分类'''
from numpy import *

def loadDataMat(filename):
	fr=open(filename)
	data=[]
	label=[]
	for line in fr.readlines():
		curLine=[]
		lineArr=line.strip().split('\t')
		for i in range(len(lineArr)-1):
			curLine.append(float(lineArr[i]))
		data.append(curLine)
		label.append(float(lineArr[-1]))
	return mat(data),label
dataMat,classLabels=loadDataMat('E:\\horseColicTraining2.txt')

def autoNorm(dataMat):  #利用最大最小进行均值归一化
	minVals=dataMat.min(0)
	maxVals=dataMat.max(0)
	ranges=maxVals-minVals
	m=dataMat.shape[0]
	normDataSet=dataMat-tile(minVals,(m,1))
	normDataSet=normDataSet/tile(ranges,(m,1))
	return normDataSet

dataMat=autoNorm(dataMat)

def loadSimpData():
	dataMat=mat([[1.,2.1],[2.,1.1],[1.3,1.],[1.,1.],[2,1]])
	classLabels=[1,1,-1,-1,1]
	return dataMat,mat(classLabels).T
#dataMat,classLabels=loadSimpData()

#开始构建单层决策树生成函数
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#dimen指的是特征，threshIneq指不等式
	retArray=ones((shape(dataMatrix)[0],1))
	if threshIneq=='lt':
		retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
	else: 
		retArray[dataMatrix[:,dimen]>threshVal]=-1.0
	return retArray
def build(dataMat,classLabels,D): #D样本权重
	step=10
	m,n=dataMat.shape
	classLabels=mat(classLabels).T
	bestStump={}  #储存决策树
	bestClassEst=mat(zeros((m,1)))  #储存决策树分类结果
	minError=inf
	for i in range(n):
		rangeMin=dataMat[:,i].min()
		rangeMax=dataMat[:,i].max()
		stepSize=(rangeMax-rangeMin)/step
		for j in range(-1,step):
			for inequal in ['lt','gt']:
				threshVal=(rangeMin+float(j)*stepSize)
				predict=stumpClassify(dataMat,i,threshVal,inequal)
				errArr=mat(ones((m,1)))
				#print 'que: ',classLabels.shape,mat(predict).shape
				errArr[predict==classLabels]=0
				weightedError=sum(D.T*errArr)
				if weightedError<minError:
					minError=weightedError
					bestClassEst=predict.copy()
					bestStump['dim']=i
					bestStump['thresh']=threshVal
					bestStump['ineq']=inequal
	return bestStump,minError,bestClassEst
	
#基于单层决策树的ADABOOST训练过程
def adaBoostTrains(dataMat,classLabels,numIt=40):
	weakClassArr=[]#储存所有分类器数组
	m=shape(dataMat)[0]
	D=mat(ones((m,1))/m)#训练数据分配权重
	aggClassEst=mat(zeros((m,1)))#保存强分类器分类结果
	for i in range(numIt):
		bestStump,error,classEst=build(dataMat,classLabels,D)
		#print "D:",D.T
		alpha=float(0.5*log((1.0-error)/max(error,1e-16))) 
		
		bestStump['alpha']=alpha
		weakClassArr.append(bestStump)#保存分类器
		#print "classEst:",classEst.T
		p=multiply(-1*alpha*mat(classLabels).T,classEst)
		p=exp(p)
		D=multiply(D,p)/D.sum()
		aggClassEst+=classEst*alpha
		#print 'aggClassEst: ',aggClassEst.T
		aggErrors=multiply(ones((m,1)),sign(aggClassEst)!=mat(classLabels).T)
		erroRate=aggErrors.sum()/m
		print "total rate:",erroRate,'\n'
		if erroRate==0.0: break
	return weakClassArr,aggClassEst #返回所有分类器

classify,aggClassEst=adaBoostTrains(dataMat,classLabels,10)
#print aggClassEst.T,dataMat.shape


def ROC(pre,classLabels):
	import matplotlib.pyplot as plt
	cur=(1.0,1.0)
	ysum=0.0
	numPosClas=sum(array(classLabels)==1.0)
	ystep=1/float(numPosClas)
	xstep=1/float(len(classLabels)-numPosClas)
	sortedIndicies=pre.argsort()
	fig=plt.figure()
	fig.clf()
	ax=plt.subplot(111)
	print sortedIndicies
	for index in sortedIndicies.tolist()[0]:
		if classLabels[index]==1.0:
			delX=0;delY=ystep
		else:
			delX=xstep;delY=0
			ysum+=cur[1]
		ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY],c='b')
		cur=(cur[0]-delX,cur[1]-delY)
	ax.plot([0,1],[0,1],'b--')
	plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate')
	plt.title('ROC curve')
	ax.axis([0,1,0,1])
	plt.show()
	print 'area:',ySum*xstep
ROC(aggClassEst.T,classLabels,)
			
'''# 分类函数，通过输入的阈值参数进行分类
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):

    # 初始化分类结果为1
    retArray = ones((shape(dataMatrix)[0],1))

    # 判断阈值中条件
    if threshIneq == 'lt':
        # 条件为小于等于阈值的类别为-1
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

# 单层决策树生成函数，找到并建立数据集上的最佳的单层决策树
def loadDataMat(filename):
	fr=open(filename)
	data=[]
	label=[]
	for line in fr.readlines():
		curLine=[]
		lineArr=line.strip().split('\t')
		for i in range(len(lineArr)-1):
			curLine.append(float(lineArr[i]))
		data.append(curLine)
		label.append(float(lineArr[-1]))
	return mat(data),label
dataMat,classLabels=loadDataMat('E:\\horseColicTraining2.txt')

def autoNorm(dataMat):  #利用最大最小进行均值归一化
	minVals=dataMat.min(0)
	maxVals=dataMat.max(0)
	ranges=maxVals-minVals
	m=dataMat.shape[0]
	normDataSet=dataMat-tile(minVals,(m,1))
	normDataSet=normDataSet/tile(ranges,(m,1))
	return normDataSet

dataMat=autoNorm(dataMat)
		


def buildStump(dataArr,classLabels,D):
    # 构建输入数据矩阵和类别标签矩阵
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)

    # 创建bestStump用于存储最佳单层决策树的相关信息
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))

    # 初始化错误率为无穷大
    minError = inf

    # 循环遍历数据集的所有特征
    for i in range(n):
        # 通过计算数据最大最小值来获取步长
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps

        # 在当前特征对应的数据上进行遍历
        for j in range(-1,int(numSteps)+1):

            # 再大于和小于之间切换不等式条件
            for inequal in ['lt', 'gt']:

                # 得到本次执行的阈值
                threshVal = (rangeMin + float(j) * stepSize)

                # 获得数据分类结果
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)

                # 计算错误率
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr

                # 判断加权错误率是否小于当前的最小错误率
                if weightedError < minError:
                    # 更新决策树
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    # 返回最佳单层决策树
    return bestStump,minError,bestClasEst

# AdaBoost训练过程，返回训练结果为弱分类器
def adaBoostTrainDS(dataArr,classLabels,numIt=40):

    # 初始化要返回的弱分类器列表
    weakClassArr = []
    m = shape(dataArr)[0]

    # 初始化D值，赋予相等的权重
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))

    # 开始训练迭代
    for i in range(numIt):

        # 获取最佳单层决策树，最小错误率和估计的类别向量
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)

        # 计算alpha值，为本次分类器输出结果的权重
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha

        # 将本次获得的决策树存入分类器列表
        weakClassArr.append(bestStump)

        # 计算下一次迭代的D值
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) 
        D = multiply(D,exp(expon))
        D = D/D.sum()

        # 计算总分类器的错误率，aggClassEst保存运行时的类型估计值
        aggClassEst += alpha*classEst
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print "total error: ",errorRate

        # 如果错误率已经为0则退出迭代
        if errorRate == 0.0: break

    # 返回弱分类器
    return weakClassArr
	
	
#print dataMat[0:10]
#print classLabels[0:10]
classify=adaBoostTrainDS(dataMat,classLabels,100)	
	
	
# AdaBoost分类函数，根据训练得到的弱分类器对数据进行分类
def adaClassify(datToClass,classifierArr):
    # 初始化输入数据
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]

    # 初始化aggClassEst为全0向量
    aggClassEst = mat(zeros((m,1)))

    # 依次使用每个弱分类器
    for i in range(len(classifierArr)):

        # 使用弱分类器得到类别估计值
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])

        # 输出的类别乘以权重后累加到aggClassEst上
        aggClassEst += classifierArr[i]['alpha']*classEst

    # 返回sign()函数处理过的aggClassEst值即为预计的类别
    return sign(aggClassEst)

# 测试算法
def adaTest():
    # 读取训练数据
    trainingSet, trainingLabels = loadDataSet('horseColicTraining2.txt')

    # 训练算法
    classifierArray = adaBoostTrainDS(trainingSet, trainingLabels)

    # 测试条目数量
    numTestVec = 67.0

    # 读取测试数据
    testSet, testLabels = loadDataSet('horseColicTest2.txt')

    # 使用分类器进行预测
    prediction10 = adaClassify(testSet, classifierArray)

    # 对比分类预测结果及真实结果并记录错误率
    errArr = mat(ones((67,1)))

    # 矩阵计算得到判断出错的条目数
    errorCount = errArr[prediction10!=mat(testLabels).T].sum()

    # 计算并输出错误率
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate
'''

	

				
	
	
