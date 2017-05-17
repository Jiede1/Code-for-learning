#-*- coding=utf8 -*-

from __future__ import division
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import random


# feedforward computing
def feedforward(w,a,x):
    f = lambda s: 1 / (1 + np.exp(-s)) 
    
    # concatenate the matrix a and x , and multiply with weight matrix
    w = np.array(w)
    temp = np.array(np.concatenate((a,x),axis=0))
    z_next = np.dot(w , temp)
    
    return f(z_next), z_next

# backpropagation
def backprop(w,z,delta_next):

    # sigmoid function
    f = lambda s: np.array(1 / (1 + np.exp(-s)))
    

    # the Derivative of sigmoid function
    df = lambda s: f(s) * (1 - f(s))
    
    
    delta = df(z) * np.dot(w.T,delta_next)    

    return delta

# autoencoder realization    

DataSet = scio.loadmat('yaleB_face_dataset.mat')
unlabeledData = DataSet['unlabeled_data']
unlabeledData = unlabeledData[:,:] / 255.

fig1 = plt.figure(1)


# define the learning parameters
alpha = 0.5 # learning rate
max_epoch = 300 # the learning epoch
mini_batch = 10 # used for the batch learning
height = 48
width = 42
imgSize = height * width # image size of a face image in the yale B+ dataset
dataset_size = 80  # the number of the images in the training data set
unlabeled_data = np.zeros(unlabeledData.shape)
for i in range(dataset_size):
    tmp = unlabeledData[:,i]
    unlabeled_data[:,i] = (tmp - np.mean(tmp)) / np.std(tmp)    
# the network structure
hidden_node = 60
hidden_layer = 2 
layer_struc = [[imgSize, 1],
               [0, hidden_node],
               [0, imgSize]]
layer_num = 3

# initialize weights
w = []
for l in range(layer_num-1):
    w.append(np.random.randn(layer_struc[l+1][1],sum(layer_struc[l])))

# define the internal input of the network
X = []
X.append(np.array(unlabeled_data[:,:]))
X.append(np.zeros((0,dataset_size)))
X.append(np.zeros((0,dataset_size)))

delta = []
for l in range(layer_num):
    delta.append([])

# define the display parameters 
nRow = max_epoch / 100 + 1
nColumn = 4   # display 10 images in each row 
eachDigitNum = 20  # 50 instancese corresponding to each digit in the training set

# display the original digit in the first row    
for iImg in range(nColumn):
    ax = plt.subplot(nRow, nColumn, iImg+1)
    plt.imshow(unlabeledData[:,eachDigitNum * iImg + 1].reshape((width,height)).T, cmap= plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# unsupervised training
count = 0 # count the iteration
print('Autoencoder training start..')
for iter in range(max_epoch):

    # define the shuffle index
    ind = list(range(dataset_size))
    random.shuffle(ind)
    
    a = []
    z = []
    z.append([])
    for i in range(int(np.ceil(dataset_size / mini_batch))):
        a.append(np.zeros((layer_struc[0][1], mini_batch)))
        x = []
        for l in range(layer_num):
            x.append( X[l][:,ind[i*mini_batch : min((i+1)*mini_batch, dataset_size)]])

        y = unlabeled_data[:,ind[i*mini_batch:min((i+1)*mini_batch,dataset_size)]]
        for l in range(layer_num-1):
            a.append([])
            z.append([])
            a[l+1],z[l+1] = feedforward(w[l],a[l],x[l])
        
        
        delta[layer_num-1] = np.array(a[layer_num-1] - y) * np.array(a[layer_num-1])
        delta[layer_num-1] = delta[layer_num-1] * np.array(1-a[layer_num-1])
        
        for l in range(layer_num-2, 0, -1):
            delta[l] = backprop(w[l],z[l],delta[l+1])

        for l in range(layer_num-1):
            dw = np.dot(delta[l+1], np.concatenate((a[l],x[l]),axis=0).T) / mini_batch
            w[l] = w[l] - alpha * dw
   
    count = count + 1  
     
    
   
    # display reconstruction result 
    if np.mod(iter+1,100) == 0 :
        b = []
        b.append(np.zeros((layer_struc[0][1],dataset_size)))

        for l in range(layer_num-1):
            tempA, tempZ = feedforward(w[l], b[l], X[l])                
            b.append(tempA)

        for iImg in range(nColumn):
            fig1
            ax = plt.subplot(nRow,nColumn, iImg + nColumn * (iter+1)/100 + 1)
            tmp = b[layer_num-1][:,eachDigitNum * iImg + 1]
            dis_result = ((tmp * np.std(tmp)) + np.mean(tmp)).reshape(width,height).T
            plt.imshow(dis_result,cmap= plt.cm.gray) 
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
                        
        print('Learning epoch:', count, '/', max_epoch)
        
fig2 = plt.figure(2)

code_result, tempZ = feedforward(w[0], b[0], X[0])

for iImg in range(nColumn):
    ax = plt.subplot(2, nColumn, iImg+1)
    plt.imshow(unlabeled_data[:,eachDigitNum * iImg + 1].reshape((width,height)).T, cmap= plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

for iImg in range(nColumn):
    ax = plt.subplot(2,nColumn,iImg+nColumn+1)
    plt.imshow(code_result[:,eachDigitNum * iImg + 1].reshape((hidden_node,1)), cmap=plt.cm.gray)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    

# classifier learning
def supervised_training():
    
    global SJ
    global SAcc
    
    flag = True
    print('Supervised training start..')
    for iter in range(max_epoch):
        
        # define the shuffle index
        ind = list(range(trainData_size))
        random.shuffle(ind)
        
        a = []
        z = []
        z.append([])
        for i in range(int(np.ceil(trainData_size / mini_batch))):
            a.append(small_trainData[:,ind[i*mini_batch:min((i+1)*mini_batch,trainData_size)]])
            x = []
            for l in range(SL_layer_num):
                x.append( np.zeros((0,min((i+1)*mini_batch,trainData_size)-i*mini_batch)))
    
            y = train_labels[:,ind[i*mini_batch:min((i+1)*mini_batch,trainData_size)]]
            for l in range(SL_layer_num-1):
                a.append([])
                z.append([])
                a[l+1],z[l+1] = feedforward(supervised_weight[l],a[l],x[l])
            
            
            delta[SL_layer_num-1] = np.array(a[SL_layer_num-1] - y) * np.array(a[SL_layer_num-1])
            delta[SL_layer_num-1] = delta[SL_layer_num-1] * np.array(1-a[SL_layer_num-1])
            
            for l in range(SL_layer_num-2, 0, -1):
                delta[l] = backprop(supervised_weight[l],z[l],delta[l+1])
    
            for l in range(SL_layer_num-1):
                dw = np.dot(delta[l+1], np.concatenate((a[l],x[l]),axis=0).T) / mini_batch
                supervised_weight[l] = supervised_weight[l] - supervised_alpha * dw
    
            tmpResult = a[SL_layer_num-1]
            SJ.append(np.sum(np.multiply(tmpResult[:] - y[:], tmpResult[:] - y[:]))/2/mini_batch)   
            SAcc.append(float(sum(np.argmax(y, axis=0) == np.argmax(tmpResult, axis=0))/mini_batch))
    
    print('Supervised learning done!')         
    plt.figure()
    plt.plot(SJ)
    plt.title('loss function')
    plt.figure()
    plt.plot(SAcc)
    plt.title('Accuracy')
        
def supervised_testing():
    print('Testing..')  
    
    
    tmpA, tmpZ = feedforward(supervised_weight[0], small_trainData, np.zeros((0,trainData_size)))    
    train_pred = np.argmax(tmpA, axis=0)
    train_res = np.argmax(train_labels,axis=0)

    train_acc = float(sum(train_pred == train_res) / trainData_size) * 100
    print('Training accuracy:%.2f%c' % (train_acc,'%'))
    
    tmpA, tmpZ = feedforward(supervised_weight[0], small_testData, np.zeros((0,testData_size)))    
    test_pred = np.argmax(tmpA, axis=0)
    test_res = np.argmax(test_labels, axis=0)
    test_acc = float(sum(test_pred == test_res) / testData_size) * 100
    print('Testing accuracy:%.2f%c' % (test_acc, '%'))
    
# initial parameter

supervised_alpha= 0.5
max_epoch = 200
mini_batch = 14
SJ = []
SAcc = []

# initial the supervised learning network structure
SL_layer_srtuc = []
SL_layer_num = 2
SL_layer_struc = [[0, hidden_node],
                  [0, 4]]

supervised_weight = []
for l in range(SL_layer_num-1):
    supervised_weight.append(np.random.randn(SL_layer_struc[l+1][1],sum(SL_layer_struc[l])))

# preparing supervised learning data
trainData = DataSet['trainData'] 
trainData = trainData[:,:]   
train_data = np.zeros(trainData.shape)

# normalization
trainData_size = 56  
for i in range(trainData_size):
    tmp = trainData[:,i] /255.
    train_data[:,i] = (tmp - np.mean(tmp)) / np.std(tmp) 
      
train_labels = DataSet['train_labels']
train_labels = train_labels[:,:]

      
testData = DataSet['testData']
testData = testData[:,:]
test_data = np.zeros(testData.shape)

# normalization 
testData_size = 40  
for i in range(testData_size):
    tmp = testData[:,i] / 255.
    test_data[:,i] = (tmp - np.mean(tmp)) / np.std(tmp) 
    
test_labels = DataSet['test_labels']
test_labels = test_labels[:,:]
   

delta = []
for l in range(SL_layer_num):
    delta.append([])         
        
# dimension reduction based on the unsupervised learning result
a = []
a.append(np.zeros((layer_struc[0][1],trainData_size)))
for l in range(hidden_layer-1):
    if l == 0:
        tmpA,tmpZ = feedforward(w[l], a[l], train_data)
        a.append(tmpA)
    else:
        tmpA,tmpZ = feedforward(w[l],a[l], np.zeros((0,trainData_size)))
        a.append(tmpA)

small_trainData = a[hidden_layer-1]

a = []
a.append(np.zeros((layer_struc[0][1],testData_size)))
for l in range(hidden_layer-1):
    if l == 0:
        tmpA,tmpZ = feedforward(w[l], a[l], test_data)
        a.append(tmpA)
    else:
        tmpA,tmpZ = feedforward(w[l],a[l], np.zeros((0,testData_size)))
    
    small_testData = a[hidden_layer-1]

supervised_training()
supervised_testing()

plt.show()
