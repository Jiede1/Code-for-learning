#-*- coding=utf8 -*-

import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import random


def main():
    
    trainData = scio.loadmat(r'D:\python_learning_dataes\trainData.mat')

    unlabeled_data = trainData['trainData']
    unlabeled_data = unlabeled_data[:,:] / 255.

    
    # define the learning parameters
    alpha = 5 # learning rate
    max_epoch = 500 # the learning epoch
    mini_batch = 100 # used for the batch learning
    imgSize = 784 # image size of a digit image in the minist dataset

    # the network structure
    layer_struc = [[imgSize, 1],
                   [0, 32],
                   [0, imgSize]]
    layer_num = 3

    # initialize weights
    w = []
    for l in range(layer_num-1):
        w.append(np.random.randn(layer_struc[l+1][1],sum(layer_struc[l])))

    dataset_size = 500  # the number of the images in the training data set
    
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
    nColumn = 10   # display 10 images in each row 
    eachDigitNum = 50  # 50 instancese corresponding to each digit in the training set

    # display the original digit in the first row

    for iImg in range(nColumn):
        ax = plt.subplot(nRow, nColumn, iImg+1)
        plt.imshow(unlabeled_data[:,eachDigitNum * iImg + 1].reshape((28,28)).T, cmap= plt.cm.gray)
       
        if iImg == 0:
            plt.ylabel('Original Images',rotation=90)

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
                ax = plt.subplot(nRow,nColumn, iImg + nColumn * (iter+1)/100 + 1)
                dis_result = b[layer_num-1][:,eachDigitNum * iImg + 1].reshape(28,28).T
                plt.imshow(dis_result,cmap= plt.cm.gray) 
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                            
            print('Learning epoch:', count, '/', max_epoch)
    
    plt.show()


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

if __name__ == '__main__':
    main()
