#-*- coding: utf-8 -*-

'''
Created on Jan 8, 2011

@author: Peter
'''

from numpy import *

# 读取地理坐标数据
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        dataMat.append([float(curLine[4]), float(curLine[3])])
    return mat(dataMat)

# 构建K个随机质心集合
def randCent(dataSet, k):
    # 初始化质心矩阵
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))

    # 遍历数据集的每一维
    for j in range(n):
        # 得到最小值
        minJ = min(dataSet[:,j])
        # 得到最大最小值之间的区间
        rangeJ = float(max(dataSet[:,j]) - minJ)
        # 调用随机函数获得随机质心
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

# 根据输入的经纬度计算两地间距离
def distSLC(vecA, vecB):
    # 使用球面余弦定理计算距离
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0

# K-均值聚类算法实现
def kMeans(dataSet, k, distMeas=distSLC, createCent=randCent):

    # 根据数据集进行初始化
    m = shape(dataSet)[0]

    # 存储每个点的簇分配结果，包括簇索引值和误差
    clusterAssment = mat(zeros((m,2)))

    # 随机产生k个质心
    centroids = createCent(dataSet, k)

    # 循环控制标志，簇分配是否有变化
    clusterChanged = True

    # 循环直到簇不再变化
    while clusterChanged:
        clusterChanged = False

        # 遍历所有数据点
        for i in range(m):
            minDist = inf; minIndex = -1

            # 遍历随机产生的质心，数据点被分配到最近的质心
            for j in range(k):
                # 计算质心与数据点的距离，寻找最近的质心
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print i
        print "centroids:",centroids

        # 重新计算质心
        for cent in range(k):
            # 通过数组过滤获得给定点簇的所有点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]

            # 计算所有点的均值，axis=0表示在列方向进行均值计算
            centroids[cent,:] = mean(ptsInClust, axis=0)

    # 迭代结束后返回类质心和数据点分配结果矩阵
    return centroids, clusterAssment

# 二分 K-均值算法
def biKmeans(dataSet, k, distMeas=distSLC):
    # 根据数据集进行初始化
    m = shape(dataSet)[0]

    # 存储每个点的簇分配结果，包括簇索引值和误差
    clusterAssment = mat(zeros((m,2)))

    # 通过均值获得第一个质心
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0]

    # 计算一个簇时的误差
    for j in range(m):
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2

    # 如果簇数量小于k则继续迭代
    while (len(centList) < k):

        lowestSSE = inf

        # 遍历每个簇
        for i in range(len(centList)):

            # 对该簇进行K-均值聚类（k=2）
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)

            # 计算SSE
            sseSplit = sum(splitClustAss[:,1])

            # 剩余数据集的误差
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print "sseSplit, and notSplit: ",sseSplit,sseNotSplit

            # 对比本次划分的误差与最小SSE
            if (sseSplit + sseNotSplit) < lowestSSE:
                # 保存本次划分
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                # 更新最小SSE
                lowestSSE = sseSplit + sseNotSplit

        # 修改簇的编号为划分簇和新加簇的编号
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print 'the bestCentToSplit is: ',bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)

        # 重新质心添加到centList
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
        centList.append(bestNewCents[1,:].tolist()[0])

        # 保存新的簇分配结果和SSE
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss

    # 迭代结束，返回类质心和数据点分配结果矩阵
    print mat(centList)
    print clusterAssment
    return mat(centList), clusterAssment

# 引入 matplotlib 进行画图
import matplotlib
import matplotlib.pyplot as plt

# 测试地理坐标聚类函数，参数为希望得到的簇的数目
def clusterClubs(numClust=5):

    # 读取地理坐标数据
    datMat = loadDataSet('F:\lab13\places.txt')

    # 执行二分K-均值聚类算法
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)

    # 开始在地图上显示簇和簇的质心
    # 初始化 Figure 对象
    fig = plt.figure()

    # 绘制一个矩形
    rect=[0.1,0.1,0.8,0.8]

    # 创建标记形状的列表(p五角星，*星星，^三角，s正方形，o圆形)
    scatterMarkers=['s', 'o', '^', '*', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)

    # 使用 imread() 绘图，读取 Portland 地图
    imgP = plt.imread('F:lab13\Portland.png')
    ax0.imshow(imgP)

    # 在同一幅图上绘制一张新图
    ax1=fig.add_axes(rect, label='ax1', frameon=False)

    # 遍历每个簇
    for i in range(numClust):

        # 在新图上绘制每个簇并进行标记
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)

    # 绘制簇的质心
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()

clusterClubs(5)