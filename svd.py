import numpy as np
from numpy import *
from numpy import linalg as la

def loadDataSet():
	return[[2,0,0,4,4,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,0,5],
			[0,0,0,0,0,0,0,1,0,4,0],
			[3,3,4,0,3,0,0,2,2,0,0],
			[5,5,5,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,5,0,0,5,0],
			[4,0,4,0,0,0,0,0,0,0,5],
			[0,0,0,0,0,0,5,0,0,5,0],
			[0,0,0,3,0,0,0,0,4,5,0],
			[1,1,2,1,1,2,1,0,4,5,0]]
	#return [[2,0,0,4,4],[5,0,5,3,3],[2,0,2,1,2]]

data=loadDataSet()
print data
U,Sigma,VT=la.svd(np.mat(data))
print U.shape,Sigma.shape,VT.shape

#similarity
def sulidSim(inA,inB):
	return 1.0/(1.0+la.norm(inA-inB))
def poearsSim(inA,inB):
	if len(inA)<3:return 1.0
	return 0.5+0.5*np.corrcoef(inA,inB,rowvar=0)[0][1]
def cosSim(inA,inB):
	num=float(inA.T*inB)
	denom=la.norm(inA)*la.norm(inB)
	return 0.5+0.5*(num/denom)

def standEst(dataMat,user,simMeas,item):
	n=dataMat.shape[1]
	simTotal=0.0;ratSimTotal=0.0
	for j in range(n):
		userRating=dataMat[user,j]
		if dataMat[user,j]==0:continue
		overlap=nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]
		print 'overlap:',overlap
		if len(overlap)==0:similarity=0
		else:similarity=simMeas(dataMat[overlap,item],dataMat[overlap,j])
		print 'dataMat[overlap,item],dataMat[overlap,j]:',dataMat[overlap,item],dataMat[overlap,j]
		simTotal+=similarity
		ratSimTotal+=similarity*userRating
	if simTotal==0:return 0
	else:return ratSimTotal/simTotal
def recommend(dataMat,user,N=3,simMeas=poearsSim,estMethod=standEst):
	unratedItems=nonzero(dataMat[user,:].A==0)[1]
	if len(unratedItems)==0:return 'you rated everything'
	itemScores=[]
	for item in unratedItems:
		estimatedScore=estMethod(dataMat,user,simMeas,item)
		itemScores.append((item,estimatedScore))
	return sorted(itemScores,key=lambda jj:jj[1],reverse=True)[:N]
	
#print recommend(mat(data),2,6)

def svbdEst(dataMat,user,simMeas,item):
	n=shape(dataMat)[1]
	simTotal=0.0;ratSimTotal=0.0
	U,sigma,VT=la.svd(dataMat)
	sig4=mat(eye(4)*sigma[:4])
	xform=dataMat.T*U[:,:4]*sig4.I
	print 'xform:',xform.shape,dataMat.shape
	for j in range(n):
		userRating=dataMat[user,j]
		if userRating==0 or j==item:continue
		similarity=simMeas(xform[item,:].T,xform[j,:].T)
		simTotal+=similarity
		ratSimTotal+=similarity*userRating
	if simTotal==0:return 0
	else:return ratSimTotal/simTotal
print recommend(mat(data),3,estMethod=standEst)
	