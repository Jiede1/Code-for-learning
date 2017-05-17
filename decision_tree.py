#coding:utf-8
import operator
from math import log
import pprint

# 加载数据
def loadData(filename):
    # 打开数据文件
    fr = open(filename)

    # 读取生成列表后存入lenses列表
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]

    # 构建标签列表
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']

    return lenses, lensesLabels
	
def loadDataSet():
	dataSet=[[1,1,'yes'],[1,1,'yes'],[0,1,'no'],[0,1,'no'],[0,0,'no'],[1,0,'no']]
	labels=['no surfacing','flippers']
	return dataSet,labels

# 计算香农熵
def calcShannonEnt(dataSet):
    # 获得数据集的大小
    numEntries = len(dataSet)
    labelCounts = {}

    # 遍历数据集
    for featVec in dataSet:
        # 提取数据中的类型，本例中为医生推荐的隐形眼镜类型
        currentLabel = featVec[-1]

        # 计算该类型眼镜在数据集中出现的次数
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    # 初始化香农熵
    shannonEnt = 0.0

    # 遍历隐形眼镜的类型字典计算香农熵
    for key in labelCounts:
        # 见书P35的计算香农熵的公式
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt

# 划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []

    # 遍历数据集
    for featVec in dataSet:
        # 判断当前划分的特征对应的值是否是需要返回的值
        if featVec[axis] == value:
            # 将当前处理的数据列表featVec进行重构组合
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 获取最好的数据划分方式
def chooseBestFeatureToSplit(dataSet):
    # 用来划分的特征数量，数据集中最后一列是分类不是特征
    numFeatures = len(dataSet[0]) - 1 

    # 计算数据集的香农熵
    baseEntropy = calcShannonEnt(dataSet)

    # 初始化
    bestInfoGain = 0.0; bestFeature = -1

    # 遍历所有特征
    for i in range(numFeatures):

        # 提取该特征对应的所有值
        featList = [example[i] for example in dataSet]

        # 去重
        uniqueVals = set(featList)
        newEntropy = 0.0

        # 循环使用该特征值作为参数进行数据集划分尝试
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)

            # 对划分后的数据集计算香农熵，并进行求和操作
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)

        # 计算信息增益
        infoGain = baseEntropy - newEntropy

        # 获取最好的信息增益
        if (infoGain > bestInfoGain): 
            bestInfoGain = infoGain 
            bestFeature = i
    return bestFeature

# 构建决策树
def createTree(dataSet,labels):
    # 提取数据集中的所有分类标签
    print dataSet
    classList = [example[-1] for example in dataSet]
    print 'classList:',classList
    # 数据集的类别完全相同时则停止继续对数据集进行划分，并返回分类标签
    if classList.count(classList[0]) == len(classList): 
        return classList[0]
	

    # 如果所有的特征值都已经遍历结束
    if len(dataSet[0]) == 1:
        # 计算出现频次最多的分类标签并返回
        classCount={}
        for vote in classList:
            if vote not in classCount.keys(): classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    # 获取最好的划分特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    print bestFeat
    bestFeatLabel = labels[bestFeat]
    print bestFeatLabel

    # 以当前数据集中获取的最好的特征初始化树
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])

    # 提取该特征对应的所有特征值
    featValues = [example[bestFeat] for example in dataSet]

    # 去重，需要根据唯一的特征值进行划分分支
    uniqueVals = set(featValues)
    print 'uniqueVals:',uniqueVals
    # 划分数据集，创建树的分支，将不同的特征值对应的数据集进行递归划分
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree

dataSet,labels=loadDataSet()
myTree=createTree(dataSet,labels)
print myTree,'\n',labels

def getNumLeafs(myTree):#找叶节点
	numLeafs=0
	firstStr=myTree.keys()[0]
	secondDict=myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			numLeafs+=getNumLeafs(secondDict[key])
		else:
			numLeafs+=1
	return numLeafs
print getNumLeafs(myTree)   #2

def getTreeDepth(myTree):#找层数
	depth=0
	firstStr=myTree.keys()[0]
	secondDict=myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			depth=1+getTreeDepth(secondDict[key])
		else:
			depth=1
	return depth
print getTreeDepth(myTree)

def classify(inputTree,freeLabels,testVec):#分类
	firstStr=inputTree.keys()[0]
	classLabel=[]
	secondDict=inputTree[firstStr]
	featIndex=freeLabels.index(firstStr)
	#print freeLabels
	for key in secondDict.keys():
		if testVec[featIndex]==key:
			if type(secondDict[key]).__name__=='dict':
				classLabel=classify(secondDict[key],freeLabels,testVec)
			else:
				classLabel=secondDict[key]
	return  classLabel

labels=['no surfacing','flippers']
print classify(myTree,labels,[1,1])

	

	
	

#dataFile = '/home/shiyanlou/mylab4/lenses.txt'
#lenses, lensesLabels = loadData(dataFile)
#tree = createTree(lenses, lensesLabels)
#pprint.pprint(tree)