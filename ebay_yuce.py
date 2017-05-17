#coding:utf-8
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import (manifold, decomposition, random_projection)
from matplotlib import offsetbox
from time import time
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDRegressor


test_set = pd.read_csv(r'D:\python_learning_dataes\raw\TestSet.csv')
train_set = pd.read_csv(r'D:\python_learning_dataes\raw\TrainingSet.csv')
test_subset = pd.read_csv(r'D:\python_learning_dataes\raw\TestSubset.csv')
train_subset = pd.read_csv(r'D:\python_learning_dataes\raw\TrainingSubset.csv')

print(train_set[:3])
print train_set.info()

train = train_set.drop(['EbayID','QuantitySold','SellerName'], axis=1)
train_target = train_set['QuantitySold']

'''可视化数据，取出一部分特征，量量组成对看数据在这个2维平面上的分布情况：'''

# 获取总特征数
_,n_features = train.shape   		#特征数25个
# isSold： 拍卖成功为1， 拍卖失败为0
df = DataFrame(np.hstack((train,train_target[:, None])), columns=range(n_features) + ["isSold"])
print df
print train.shape
print train_target[:, None].shape
_ = sns.pairplot(df[:50], vars=[2,3,4,10,13], hue="isSold", size=1.5)  #vars=[2,3,4,10,13]表示选择某几个特征
plt.plot()


plt.figure(figsize=(10,10))

# 计算数据的相关性矩阵
corr = df.corr()

# 产生遮挡出热度图上三角部分的mask，因为这个热度图为对称矩阵，所以只输出下三角部分即可
mask = np.zeros_like(corr,dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# 产生热度图中对应的变化的颜色
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# 调用seanborn中的heat创建热度图
sns.heatmap(corr, mask=mask, cmap=cmap, vmax = .3,
                square=True, xticklabels=5, yticklabels=2,
                linewidths=.5, cbar_kws={"shrink":.5})

# 将yticks旋转至水平方向，方便查看
plt.yticks(rotation=0)

plt.show()

n_trainSamples, n_features = train.shape

# 画出训练过程中SGDClassifier利用不同的mini_batch学习的效果
def plot_learning(clf,title):

    plt.figure()
    # 记录上一次训练结果在本次batch上的预测情况
    validationScore = []
    # 记录加上本次batch训练结果后的预测情况
    trainScore = []
    # 最小训练批数
    mini_batch = 1000

    for idx in range(int(np.ceil(n_trainSamples / mini_batch))):
        x_batch = train[idx * mini_batch: min((idx + 1) * mini_batch, n_trainSamples)]
        y_batch = train_target[idx * mini_batch: min((idx + 1) * mini_batch, n_trainSamples)]

        if idx > 0:
            validationScore.append(clf.score(x_batch, y_batch))
        clf.partial_fit(x_batch, y_batch, classes=range(5))
        if idx > 0:
            trainScore.append(clf.score(x_batch, y_batch))
    plt.plot(trainScore, label="train score")
    plt.plot(validationScore, label="validation socre")
    plt.xlabel("Mini_batch")
    plt.ylabel("Score")
    plt.legend(loc='best')
    plt.grid()
    plt.title(title)

# 对数据进行归一化
scaler = StandardScaler()
train = scaler.fit_transform(train)
# 创建SGDClassifier
clf = SGDClassifier(penalty='l2', alpha=0.001)
plot_learning(clf,"SGDClassifier")

plt.show()

# image[0]为数字0表示某条数据的isSold为0
images= []
images.append([[  0. ,   0. ,   5. ,  13. ,   9. ,   1. ,   0. ,   0. ],
 [  0. ,   0. ,  13. ,  15. ,  10. ,  15. ,   5. ,   0. ],
 [  0. ,   3. ,  15. ,   2. ,   0. ,  11. ,   8. ,   0. ],
 [  0. ,   4. ,  12. ,   0. ,   0. ,   8. ,   8. ,   0. ],
 [  0. ,   5. ,   8. ,   0. ,   0. ,   9. ,   8. ,   0. ],
 [  0. ,   4. ,  11. ,   0. ,   1. ,  12. ,   7. ,   0. ],
 [  0. ,   2. ,  14. ,   5. ,  10. ,  12. ,   0. ,   0. ],
 [  0. ,   0. ,   6. ,  13. ,  10. ,   0. ,   0. ,   0. ]])
# image[1]为数字1表示某条数据的isSold为1
images.append([[  0. ,   0. ,   0. ,  12. ,  13. ,   5. ,   0. ,   0. ],
 [  0. ,   0. ,   0. ,  11. ,  16. ,   9. ,   0. ,   0. ],
 [  0. ,   0. ,   3. ,  15. ,  16. ,   6. ,   0. ,   0. ],
 [  0. ,   7. ,  15. ,  16. ,  16. ,   2. ,   0. ,   0. ],
 [  0. ,   0. ,   1. ,  16. ,  16. ,   3. ,   0. ,   0. ],
 [  0. ,   0. ,   1. ,  16. ,  16. ,   6. ,   0. ,   0. ],
 [  0. ,   0. ,   1. ,  16. ,  16. ,   6. ,   0. ,   0. ],
 [  0. ,   0. ,   0. ,  11. ,  16. ,  10. ,   0. ,   0. ]])

# 这里只选取了1000条数据进行数据可视化展示
show_instancees = 1000
# define the drawing function
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i,0], X[i,1], str(train_target[i]),
                 color=plt.cm.Set1(train_target[i] / 2.),
                 fontdict={'weight':'bold','size':9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        shown_images = np.array([[1., 1.]])  # just something big
    for i in range(show_instancees):
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-3:
            # don't show points that are too close
            continue
        shown_images = np.r_[shown_images, [X[i]]]
        auctionbox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(images[train_target[i]], cmap=plt.cm.gray_r), X[i])
        ax.add_artist(auctionbox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

# 随机投影 Random Projection
start_time = time()
rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
rp.fit(train[:show_instancees])
train_projected = rp.transform(train[:show_instancees])
plot_embedding(train_projected, "Random Projection of the auction (time: %.3fs" % (time() - start_time))

# 主成分提取 PCA
start_time = time()
train_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(train[:show_instancees])
plot_embedding(train_pca, "Pricincipal components projection of the auction (time: %.3fs" % (time() - start_time))

# t-sne 降维
start_time = time()
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
train_tsne = tsne.fit_transform(train[:show_instancees])
plot_embedding(train_tsne, "T-SNE embedding of the auction (time: %.3fs" % (time() - start_time))


# 导入相关包
from sklearn.metrics import (precision_score, recall_score, f1_score)
# 首先也是准备好测试数据，进行归一化处理
test = test_set.drop(['EbayID','QuantitySold','SellerName'],axis=1)
test_target = test_set['QuantitySold']
test = scaler.fit_transform(test)

# 利用训练好的分类器进行预测
test_pred = clf.predict(test) 

print("SGDClassifier training performance on testing dataset:" )
print("\tPrecision: %1.3f " % precision_score(test_target, test_pred))
print("\tRecall: %1.3f" % recall_score(test_target, test_pred))
print("\tF1: %1.3f \n" % f1_score(test_target, test_pred))

# prepare data
test_subset = pd.read_csv(r'D:\python_learning_dataes\raw\TestSubset.csv')
train_subset = pd.read_csv(r'D:\python_learning_dataes\raw\TrainingSubset.csv')

# 训练集
train = train_subset.drop(['EbayID','Price','SellerName'],axis=1)
train_target = train_subset['Price']

scaler = MinMaxScaler()
train = scaler.fit_transform(train)
n_trainSamples, n_features = train.shape

# ploting example from scikit-learn
def plot_learning(clf,title):

    plt.figure()
    validationScore = []
    trainScore = []
    mini_batch = 500
    # define the shuffle index
    ind = list(range(n_trainSamples))
    random.shuffle(ind)

    for idx in range(int(np.ceil(n_trainSamples / mini_batch))):
        x_batch = train[ind[idx * mini_batch: min((idx + 1) * mini_batch, n_trainSamples)]]
        y_batch = train_target[ind[idx * mini_batch: min((idx + 1) * mini_batch, n_trainSamples)]]

        if idx > 0:
            validationScore.append(clf.score(x_batch, y_batch))
        clf.partial_fit(x_batch, y_batch)
        if idx > 0:
            trainScore.append(clf.score(x_batch, y_batch))

    plt.plot(trainScore, label="train score")
    plt.plot(validationScore, label="validation socre")
    plt.xlabel("Mini_batch")
    plt.ylabel("Score")
    plt.legend(loc='best')
    plt.title(title)

sgd_regresor = SGDRegressor(penalty='l2',alpha=0.001)
plot_learning(sgd_regresor,"SGDRegressor")

# 准备测试集查看测试情况
test = test_subset.drop(['EbayID','Price','SellerName'],axis=1)
test = scaler.fit_transform(test)
test_target = test_subset['Price']

print("SGD regressor prediction result on testing data: %.3f" % sgd_regresor.score(test,test_target))

plt.show()


