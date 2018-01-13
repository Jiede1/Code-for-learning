#coding:utf-8
import display_network
import scipy.io
import numpy as np

patches = scipy.io.loadmat('E:/important_dataset/stlSampledPatches.mat')['patches']
print("patches: ",patches.shape)

#display_network.display_color_network(patches[:, 0:100], filename='patches_raw.png')


def element_wise_op(array, op):
    c=np.ones(shape=array.shape)
    k=0
    op=op.flatten()
    for i in np.nditer(array,
                       op_flags=['readwrite']):
        print(i)
        c[k]=i*op[k]
        k+=1
    return c

a=np.array([2,2,3,4])
b=np.array([2,3,4,5])
print(element_wise_op(a,b))

c=np.zeros((3,5,6))
print(c.shape[-2])
print(c.shape[-1])


# 为数组增加Zero padding
def padding(input_array, zp):
    '''
    为数组增加Zero padding，自动适配输入为2D和3D的情况
    '''
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            padded_array = np.zeros((
                input_depth,
                input_height + 2 * zp,
                input_width + 2 * zp))
            padded_array[:,
                zp : zp + input_height,
                zp : zp + input_width] = input_array
            return padded_array
        elif input_array.ndim == 2:
            input_width = input_array.shape[1]
            input_height = input_array.shape[0]
            padded_array = np.zeros((
                input_height + 2 * zp,
                input_width + 2 * zp))
            padded_array[zp : zp + input_height,
                zp : zp + input_width] = input_array
            return padded_array

input_array=np.random.random((3,4,4))
zp=1
#print(padding(input_array, zp))

stl_train = scipy.io.loadmat('E:\important_dataset\stlTrainSubset.mat')
train_images = stl_train['trainImages']
train_labels = stl_train['trainLabels']
num_train_images = stl_train['numTrainImages'][0][0]
print(train_images.shape,train_labels.shape)

