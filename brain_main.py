#coding：utf-8

#author: MingYu Pang  2017/8/1
#项目实现：利用SVM,KNN,LR,GBDT,LDA,RF,DTC算法实现脑电波身份识别
#数据集：利用三个文件，chenminyi3,suyuxiao02,yuli06三人的脑电波数据，构成(64*3，10)的数据集。
#项目之所以降维数为10，在


import os
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
import scipy.io as sio  #用于读取mat文件
from sklearn.decomposition import PCA
from sklearn. preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit


#读取sklearn自带的数据集（鸢尾花）
def getData_1():
    iris = datasets.load_iris()
    X = iris.data   #样本特征矩阵，150*4矩阵，每行一个样本，每个样本维度是4
    y = iris.target #样本类别矩阵，150维行向量，每个元素代表一个样本的类别
    return X,y

#读取cnt文件转化而来的mat文件
def getData_brain_cnt():
    liukemin01=sio.loadmat(u'E:\项目数据样本\chenminyi3.mat')['EEG']['data'][0,0]
    label_liukemin01=np.zeros(liulemin01.shape[0])
    print('liukemin01.shape,label_liukemin01.shape: ',liukemin01.shape,label_liukemin01.shape)
    
    #数据集太大，无法PCA
    #chenminyi3=sio.loadmat(u'E:\项目数据样本\chenminyi3.mat')['EEG']['data'][0,0]
    #label_chenminyi3=np.ones(chenminyi3.shape[0]).ravel()
    #print('chenminyi3.shape,label_chenminyi3.shape: ',chenminyi3.shape,label_chenminyi3.shape)	
    
    suyuxiao02=loadmat(u'E:\项目数据样本\suyuxiao02.mat') 
    label_suyuxiao02=np.ones(suyuxiao02.shape[0])
    print('suyuxiao02.shape,label_suyuxiao02.shape: ',suyuxiao02.shape,label_suyuxiao02.shape)
    
    yuli06=sio.loadmat(u'E:\项目数据样本\yuli06.mat')['EEG']['data'][0,0]
    label_yuli06=np.ones(yuli06.shape[0])+1
    print('yuli06.shape,label_yuli06.shape: ',yuli06.shape,label_yuli06.shape)


#PCA降维，k是降维数,本项目用不到
def PCA_algorthm(dataSet,classLabels,k=10):
    if k==10:
        D=np.load(u'E:\项目数据样本\D134.npz')
        chenminyi3=D['D1']
        suyuxiao02=D['D3']
        yuli06=D['D4']

    
#返回保存在E盘的数据
def getData():    
    D=np.load(u'E:\项目数据样本\D134.npz')
    chenminyi3=D['D1']
    suyuxiao02=D['D3']
    yuli06=D['D4']
    class0=np.zeros(len(chenminyi3))
    class1=np.ones(len(suyuxiao02))
    class2=np.ones(len(yuli06))+1
    
    dataSet=np.concatenate((chenminyi3,suyuxiao02,yuli06))
    #归一化
    dataSet=StandardScaler().fit_transform(dataSet)
    classLabels=np.concatenate((class0,class1,class2))
    
    #print('dataSet,classLabels:',dataSet.shape,classLabels.shape)
    
    return dataSet,classLabels

#数据集划分,划分0.6，0.4
def getData_2():
    dataSet,classLabels=getData()
    X_train1, X_test1, y_train1, y_test1 = train_test_split(dataSet,classLabels, test_size = 0.4, random_state = 8)
    return X_train1,y_train1, X_test1, y_test1
    


#【K的含义】假设一共有1000个样本，K取10，那么就将这1000个样本切分10份（一份100个），那么就产生了10个测试集
#对于每一份的测试集，剩余900个样本即作为训练集
#结果返回一个字典：键为集合编号（1train, 1trainclass, 1test, 1testclass, 2train, 2trainclass, 2test, 2testclass...），值为数据
#其中1train和1test为随机生成的第一组训练集和测试集（1trainclass和1testclass为训练样本类别和测试样本类别），其他以此类推
def getData_3():
    dataSet,classLabels=getData()
    setDict = {}    #创建字典，用于存储生成的训练集和测试集
    count = 0
    skf=StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=8)
    for train_index,test_index in skf.split(dataSet,classLabels):
        trainSet=dataSet[train_index]
        trainclass=classLabels[train_index]
        testSet=dataSet[test_index]
        testclass=classLabels[test_index]
        setDict[str(count) + 'train'] = trainSet
        setDict[str(count) + 'trainclass'] = trainclass
        setDict[str(count) + 'test'] = testSet
        setDict[str(count) + 'testclass'] = testclass
        count+=1
    return setDict

#K近邻（K Nearest Neighbor）
def KNN():
    clf = neighbors.KNeighborsClassifier()
    return clf

#线性鉴别分析（Linear Discriminant Analysis）
def LDA():
    clf = LinearDiscriminantAnalysis()
    return clf

#支持向量机（Support Vector Machine）
def SVM():
    clf = svm.SVC()
    return clf

#逻辑回归（Logistic Regression）
def LR():
    clf = LogisticRegression()
    return clf

#随机森林决策树（Random Forest）
def RF():
    clf = RandomForestClassifier()
    return clf

#多项式朴素贝叶斯分类器
def native_bayes_classifier():
    clf = MultinomialNB(alpha = 0.01)
    return clf

#决策树
def decision_tree_classifier():
    clf = tree.DecisionTreeClassifier()
    return clf

#GBDT
def gradient_boosting_classifier():
    clf = GradientBoostingClassifier(n_estimators = 20)
    return clf

def MLP():
    from sklearn.neural_network import MLPClassifier
    clf=MLPClassifier(max_iter=1500)
    return clf
    
#计算识别率
def getRecognitionRate(testPre,testClass,name='algorithm'):
    testNum = len(testPre)
    rightNum = 0
    for i in range(0, testNum):
        if testClass[i] == testPre[i]:
            rightNum += 1
    fr=open("F:/test_result.txt",'a+')
    if name!='algorithm':
        fr.write("Model {0} : {1} ".format(name,float(rightNum) / float(testNum)))
        fr.write("\n")
    return float(rightNum) / float(testNum)

#report函数，将调参的详细结果存储到本地F盘（路径可自行修改，其中n_top是指定输出前多少个最优参数组合以及该组合的模型得分）
def report(results,name, n_top=5):
    f = open('F:/params/grid_search_'+name+'.txt', 'w+')
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        print(candidates)  
        for candidate in candidates:
            f.write("Model with rank: {0}".format(i) + '\n')
            f.write("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]) + '\n')
            f.write("Parameters: {0}".format(results['params'][candidate]) + '\n')
            f.write("\n")
    f.close()

#自动调参
def selectRFParam():
    BestParams={}
    #随机森林
    clf_RF = RF()
    param_grid_RF = {"max_depth": [3,15],
                  "min_samples_split": [3, 5, 10],
                  "min_samples_leaf": [3, 5, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"],
                  "n_estimators": range(10,50,10)}
                  # "class_weight": [{0:1,1:13.24503311,2:1.315789474,3:12.42236025,4:8.163265306,5:31.25,6:4.77326969,7:19.41747573}],
                  # "max_features": range(3,10),
                  # "warm_start": [True, False],
                  # "oob_score": [True, False],
                  # "verbose": [True, False]}
    #grid_search = GridSearchCV(clf_RF, param_grid=param_grid_RF, n_jobs=4)
    #start = time()
    #T = getData_2()    #获取数据集
    #grid_search.fit(T[0], T[1]) #传入训练集矩阵和训练样本类标
    #print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
    #      % (time() - start, len(grid_search.cv_results_['params'])))
    #report(grid_search.cv_results_)
    #BestParams['randomForest']=grid_search.cv_results_
    
    #神经网络
    param_grid_MLP={
                    "hidden_layer_sizes":[(9),(6),(8),(10)],
                    "activation":['identity','logistic','tanh','relu'],
                    "solver":['lbfgs','sgd','adam'],
                    "alpha":[0.0001,0.003,0.001,0.003,0.01,0.03,0.1,0.3],
                    "learning_rate":['constant'],
                    "max_iter":[2500,3000,3500]}
    param_grid_SVM={
                    "C":[0.001,0.1,0.3,1,10],
                    "kernel":['rbf','linear','poly','sigmoid'],
                    'gamma':[0.001,0.1,1,10]
                    }
    param_grid_KNN={"n_neighbors":[5,10],
                    'weights':['uniform','distance'],
                    'algorithm':['ball_tree','kd_tree','brute']
                    }
    param_grid_GDBT={
                     "n_estimators":[100,50,150],
                      "max_depth":range(1,5),
                       "min_samples_leaf":range(1,5)
                    }
    param_grid_LDA={
                    'solver':['svd','lsqr','eigen'],
                    }
    param_grid_LR={
                  'C':[0.1,0.3,1,10],
                  'solver':['newton-cg','lbfgs','liblinear','sag'],
                  'multi_class':['ovr']}
    clf_KNN = KNN()
    clf_LDA = LDA()
    clf_SVM = SVM()
    clf_LR = LR()
    clf_RF = RF()
    #clf_NBC = native_bayes_classifier()
    clf_DTC = decision_tree_classifier()
    clf_GBDT = gradient_boosting_classifier()
    clf_MLP=MLP()
    
    param_name=[param_grid_RF,param_grid_MLP,param_grid_SVM,param_grid_KNN,param_grid_GDBT,param_grid_LDA,param_grid_LR]
    clf_all=[clf_RF,clf_MLP,clf_SVM,clf_KNN,clf_GBDT,clf_LDA,clf_LR]
    names=['RF','MLP','SVM',"KNN",'GBDT','LDA','LR']
    
    #param_name=[param_grid_GDBT,param_grid_LDA,param_grid_LR]
    #clf_all=[clf_GBDT,clf_LDA,clf_LR]
    #names=['GBDT','LDA',"LR"]
    T = getData_2()    #获取数据集
    for name,param_name_now,clf_now in zip(names,param_name,clf_all):     #遍历这几种算法
        grid_search = GridSearchCV(clf_now, param_grid=param_name_now, n_jobs=1)
        start = time()
        grid_search.fit(T[0], T[1]) #传入训练集矩阵和训练样本类标
        print("%s GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (name,time() - start, len(grid_search.cv_results_['params'])))
        report(grid_search.cv_results_,name)
        candidate = np.nonzero(grid_search.cv_results_['rank_test_score'] == 1) 
        print('candidate',candidate[0][0])
		#选出表现最好的参数的索引
        BestParams[name]=grid_search.cv_results_['params'][candidate[0][0]]
    to_dat(BestParams)  #将最好的结果保存在dat文件中
    return BestParams

#K近邻（K Nearest Neighbor）
def KNN_best(BestParams):
    clf = neighbors.KNeighborsClassifier(algorithm=BestParams['KNN']['algorithm'],n_neighbors=BestParams['KNN']['n_neighbors'],weights=BestParams['KNN']['weights'])
    return clf

#线性鉴别分析（Linear Discriminant Analysis）
def LDA_best(BestParams):
    clf = LinearDiscriminantAnalysis(solver=BestParams['LDA']['solver'])
    return clf

#支持向量机（Support Vector Machine）
def SVM_best(BestParams):
    clf = svm.SVC(C=BestParams['SVM']['C'],gamma=BestParams['SVM']['gamma'],kernel=BestParams['SVM']['kernel'],decision_function_shape='ovr')
    return clf

#逻辑回归（Logistic Regression）
def LR_best(BestParams):
    clf = LogisticRegression(C=BestParams['LR']['C'],multi_class=BestParams['LR']['multi_class'],solver=BestParams['LR']['solver'])
    return clf

#决策树
def decision_tree_classifier_best(BestParams):
    clf = tree.DecisionTreeClassifier()
    return clf
	
#随机森林决策树（Random Forest）
def RF_best(BestParams):
    clf = RandomForestClassifier(bootstrap=BestParams['RF']['bootstrap'],criterion=BestParams['RF']['criterion'],max_depth=BestParams['RF']['max_depth'],min_samples_leaf=BestParams['RF']['min_samples_leaf'],min_samples_split=BestParams['RF']['min_samples_split'],n_estimators=BestParams['RF']['n_estimators'])
    return clf
#GBDT
def gradient_boosting_classifier_best(BestParams):
    clf = GradientBoostingClassifier(n_estimators =int(BestParams['GBDT']['n_estimators']),min_samples_leaf=int(BestParams['GBDT']['min_samples_leaf']),max_depth=int(BestParams['GBDT']['max_depth']))
    return clf

def MLP_best(BestParams):
    from sklearn.neural_network import MLPClassifier
    clf=MLPClassifier(max_iter=BestParams['MLP']['max_iter'],solver=BestParams['MLP']['solver'],
    activation=BestParams['MLP']['activation'],alpha=BestParams['MLP']['alpha'],hidden_layer_sizes=BestParams['MLP']['hidden_layer_sizes'],learning_rate=BestParams['MLP']['learning_rate'])
    return clf
#“主”函数1（KFold方法生成K个训练集和测试集，即数据集采用getData_3()函数获取，计算这K个组合的平均识别率）
def totalAlgorithm_1():
    #获取各个分类器
    clf_KNN = KNN()
    clf_LDA = LDA()
    clf_SVM = SVM()
    clf_LR = LR()
    clf_RF = RF()
    #clf_NBC = native_bayes_classifier()
    clf_DTC = decision_tree_classifier()
    clf_GBDT = gradient_boosting_classifier()
    clf_MLP=MLP()
    #获取训练集和测试集
    setDict = getData_3()
    setNums = int(len(setDict.keys()) / 4)  #一共生成了setNums个训练集和setNums个测试集，它们之间是一一对应关系
    #定义变量，用于将每个分类器的所有识别率累加
    KNN_rate = 0.0
    LDA_rate = 0.0
    SVM_rate = 0.0
    LR_rate = 0.0
    RF_rate = 0.0
    MLP_rate=0.0
    #NBC_rate = 0.0
    DTC_rate = 0.0
    GBDT_rate = 0.0
    for i in range(0, setNums):
        trainMatrix = setDict[str(i) + 'train']
        trainClass = setDict[str(i) + 'trainclass']
        testMatrix = setDict[str(i) + 'test']
        testClass = setDict[str(i) + 'testclass']
        #输入训练样本
        try:
            clf_KNN.fit(trainMatrix, trainClass)
            clf_LDA.fit(trainMatrix, trainClass)
            clf_SVM.fit(trainMatrix, trainClass)
            clf_LR.fit(trainMatrix, trainClass)
            clf_RF.fit(trainMatrix, trainClass)
            #clf_NBC.fit(trainMatrix, trainClass)
            clf_DTC.fit(trainMatrix, trainClass)
            clf_GBDT.fit(trainMatrix, trainClass)
            clf_MLP.fit(trainMatrix,trainClass)
        except:
            print('出现错误')
            print(trainMatrix.shape, trainClass.shape)
            print(trainMatrix)
        #计算识别率
        KNN_rate += getRecognitionRate(clf_KNN.predict(testMatrix), testClass)
        LDA_rate += getRecognitionRate(clf_LDA.predict(testMatrix), testClass)
        SVM_rate += getRecognitionRate(clf_SVM.predict(testMatrix), testClass)
        LR_rate += getRecognitionRate(clf_LR.predict(testMatrix), testClass)
        RF_rate += getRecognitionRate(clf_RF.predict(testMatrix), testClass)
        #NBC_rate += getRecognitionRate(clf_NBC.predict(testMatrix), testClass)
        DTC_rate += getRecognitionRate(clf_DTC.predict(testMatrix), testClass)
        GBDT_rate += getRecognitionRate(clf_GBDT.predict(testMatrix), testClass)
        MLP_rate+=getRecognitionRate(clf_MLP.predict(testMatrix),testClass)
    #输出各个分类器的平均识别率（K个训练集测试集，计算平均）
    print
    print
    print
    print('K Nearest Neighbor mean recognition rate: ', KNN_rate / float(setNums))
    print('Linear Discriminant Analysis mean recognition rate: ', LDA_rate / float(setNums))
    print('Support Vector Machine mean recognition rate: ', SVM_rate / float(setNums))
    print('Logistic Regression mean recognition rate: ', LR_rate / float(setNums))
    print('Random Forest mean recognition rate: ', RF_rate / float(setNums))
    #print('Native Bayes Classifier mean recognition rate: ', NBC_rate / float(setNums))
    print('Decision Tree Classifier mean recognition rate: ', DTC_rate / float(setNums))
    print('Gradient Boosting Decision Tree mean recognition rate: ', GBDT_rate / float(setNums))
    print("MLPClassifier recognition rate: ",MLP_rate/float(setNums))

#“主”函数2（每类前x%作为训练集，剩余作为测试集，即数据集用getData_2()方法获取，计算识别率）
def totalAlgorithm_2(BestParams):
    #获取各个分类器
    clf_KNN = KNN_best(BestParams)
    clf_LDA = LDA_best(BestParams)
    clf_SVM = SVM_best(BestParams)
    clf_LR = LR_best(BestParams)
    clf_RF = RF_best(BestParams)
    #clf_NBC = native_bayes_classifier()
    clf_MLP=MLP_best(BestParams)
    clf_DTC = decision_tree_classifier_best(BestParams)
    clf_GBDT = gradient_boosting_classifier_best(BestParams)
    #获取训练集和测试集
    T = getData_2()
    trainMatrix, trainClass, testMatrix, testClass = T[0], T[1], T[2], T[3]
    #输入训练样本
    clf_KNN.fit(trainMatrix, trainClass)
    clf_LDA.fit(trainMatrix, trainClass)
    clf_SVM.fit(trainMatrix, trainClass)
    clf_LR.fit(trainMatrix, trainClass)
    clf_RF.fit(trainMatrix, trainClass)
    #clf_NBC.fit(trainMatrix, trainClass)
    clf_MLP.fit(trainMatrix,trainClass)
    clf_DTC.fit(trainMatrix, trainClass)
    clf_GBDT.fit(trainMatrix, trainClass)

    #输出各个分类器的识别率
    print('K Nearest Neighbor recognition rate: ', getRecognitionRate(clf_KNN.predict(testMatrix), testClass,'KNN'))
    print('Linear Discriminant Analysis recognition rate: ', getRecognitionRate(clf_LDA.predict(testMatrix), testClass,'LDA'))
    print('Support Vector Machine recognition rate: ', getRecognitionRate(clf_SVM.predict(testMatrix), testClass,"SVM"))
    print('Logistic Regression recognition rate: ', getRecognitionRate(clf_LR.predict(testMatrix), testClass,"LR"))
    print('Random Forest recognition rate: ', getRecognitionRate(clf_RF.predict(testMatrix), testClass,"RF"))
    #print('Native Bayes Classifier recognition rate: ', getRecognitionRate(clf_NBC.predict(testMatrix), testClass))
    print('Decision Tree Classifier recognition rate: ', getRecognitionRate(clf_DTC.predict(testMatrix), testClass,"DTC"))
    print('Gradient Boosting Decision Tree recognition rate: ', getRecognitionRate(clf_GBDT.predict(testMatrix), testClass,"GBDT"))
    print("MLPClassifier recognition rate: ",getRecognitionRate(clf_MLP.predict(testMatrix),testClass,"MLP"))
    
    names=['KNN','LDA','SVM','LR','RF','MLP','DTC','GBDT']
    clf_all=[clf_KNN,clf_LDA,clf_SVM,clf_LR,clf_RF,clf_MLP,clf_DTC,clf_GBDT]
    AUC={}
    for nm,clf_auto in zip(names,clf_all):
        #if clf_auto==clf_SVM:
        AUC[nm]=plotROC_multi(testMatrix,testClass,clf_auto,nm)

    
#只能用于二类分配
def plotROC(dataSet_test,target_test,clf,name):  
    if hasattr(clf, "decision_function"):
        predStrengths = clf.decision_function(dataSet_test)
    else:
        predStrengths = clf.predict_proba(dataSet_test)[:,1]
    classLabels=target_test    
    import matplotlib.pyplot as plt  
    cur = (1.0,1.0) #保留绘制光标的位置  
    ySum = 0.0 #计算AUC的值  
    numPosClas = sum(array(classLabels)==1.0)  
    yStep = 1/float(numPosClas);   
    xStep = 1/float(len(classLabels)-numPosClas)  
    sortedIndicies = predStrengths.argsort()#获取排序索引  
    fig = plt.figure()  
    fig.clf()  
    ax = plt.subplot(111)  
    #画图  
    #print(sortedIndicies.tolist())
    for index in sortedIndicies.tolist():  
        if classLabels[index] == 1.0:  
            delX = 0;   
            delY = yStep;  
        else:  
            delX = xStep;   
            delY = 0;  
            ySum += cur[1]  
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')  
        cur = (cur[0]-delX,cur[1]-delY)  
    ax.plot([0,1],[0,1],'b--')  
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')  
    plt.title('ROC curve for '+name)  
    ax.axis([0,1,0,1])  
    plt.show()  
    print (name," -- the Area Under the Curve is: ",ySum*xStep  )
    return ySum*xStep

#用于多类    
def plotROC_multi(dataSet_test,target_test,clf,name):
    # Binarize the output
    y = label_binarize(target_test, classes=[0, 1, 2])
    n_classes = y.shape[1]

    # Learn to predict each class against the other
    classifier = clf
    if hasattr(clf,"decision_function"):
        y_score = clf.decision_function(dataSet_test)
    else:
        y_score = clf.predict_proba(dataSet_test)

    # Compute micro-average ROC curve and ROC area
    fpr = dict()
    tpr = dict()
    roc_auc=dict()
    if y_score.shape[1]==3:
        fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name+' Receiver operating characteristic example')
    plt.legend(loc="lower right")
    u='F:\\' + name+ '.png'
    plt.savefig(u)
    plt.show()
    
    return roc_auc['micro']

def to_dat(input):
    import pandas as pd
    input_pd=pd.DataFrame(input)
    input_pd.to_pickle("F:/BestParams.dat")
def read_dat(filename):
    import pandas as pd
    fr=pd.read_pickle(filename)
    return fr
        
	
    
if __name__ == '__main__':     
    print('getData_2: K个训练集和测试集的平均识别率')
    totalAlgorithm_1()
    print
    print('getData_3：每类前x%训练，剩余测试，各个模型的识别率')
    #BestParams=selectRFParam()
    BestParams=read_dat("F:\BestParams.dat")
    totalAlgorithm_2(BestParams)
    print('参数调优完成！')