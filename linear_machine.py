from numpy import *
a=mat('-1 0 -2; 0 5 3')
def hardlim(dataset):
	dataset[nonzero(dataset.A>0)[0]]=1
	dataset[nonzero(dataset.A<=0)[0]]=0
	return dataset
print hardlim(a)