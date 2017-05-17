#coding:utf-8
from numpy import *
import matplotlib.pyplot as plt

class BPnet(object):
	def __init__(self):
		self.learningRate=0.1  #学习率
		self.learningRate_max=0.1
		self.learningRate_min=0.01
		self.iterator=0   #当前迭代次数
		self.iterator_max=2500
		self.mc=0.3    #动量因子，用于调控权值
		self.eb=0.01   #误差容限
		
		#以下属性通过其他函数生成
		
		self.nNode=4  #隐含层节点，默认为4
		self.nIn=0    #输入层节点
		self.nOut=1   #输出层节点,默认为1
		self.example=0 #样例个数
		self.errorList=[]  #误差列表
		self.dataMat=[]    #训练集
		self.classLabels=[]  #分类标签集，作为期望输出
	
	def	init_hiddenWB(self):     #初始化输入层与隐含层之间的权重向量
		self.hi_w=(random.rand(self.nNode,self.nIn)-0.5)*2
		self.hi_b=(random.rand(self.nNode,1)-0.5)*2  #对应于偏倚单位
		self.hi_wb=concatenate((self.hi_b,self.hi_w),axis=1)
	def init_OutputWB(self):     #初始化隐含层与输出层之间的权重向量
		self.oh_w=(random.rand(self.nOut,self.nNode)-0.5)*0.5
		self.oh_b=(random.rand(self.nOut,1)-0.5)*2   #对应于偏倚单位
		self.oh_wb=concatenate((self.oh_b,self.oh_w),axis=1)
		
	def loadDataMat(self,filename):
		fr=open(filename)
		for line in fr.readlines():
			lineArr=line.strip().split('\t')
			self.classLabels.append(int(lineArr[-1]))  #分类标签集，作为期望输出
			lineArr=[1]+[float(lineArr[i]) for i in range(len(lineArr)-1)] 
			self.dataMat.append(lineArr)
		self.dataMat=mat(self.dataMat)
		self.classLabels=mat(self.classLabels).T 
		m,n=shape(self.dataMat)  #这里的n已经包括了偏倚单位
		self.nIn=n-1     
		self.example=m
		
	def normalize1(self):   #数据集归一化  min-max标准化（Min-Max Normalization）
		[m,n]=shape(self.dataMat)
		for i in range(1,n):
			self.dataMat[:,i]=(self.dataMat[:,i]-min(self.dataMat[:,i]))/(max(self.dataMat[:,i])-min(self.dataMat[:,i]))
	def normalize2(self):	#数据集归一化  Z-score标准化方法
		[m,n]=shape(self.dataMat)
		for i in range(1,n):
			self.dataMat[:,i]=(self.dataMat[:,i]-mean(self.dataMat[:,i]))/(std(self.dataMat[:,i])+1.0e-10)
	
	def drawClassScatter(self):  #绘制数据集的二维离散点
		n=self.dataMat.shape[1]
		i=0
		if n==3:
			fig=plt.figure()
			ax=fig.add_subplot(111)
			for mydata in self.dataMat:
				if self.classLabels[i]==0:
					ax.scatter(mydata[0,1],mydata[0,2],c='blue',marker='o')
				if self.classLabels[i]==1:
					ax.scatter(mydata[0,1],mydata[0,2],c='red',marker='o')
				i+=1
		else:
			print 'can not plot the dataMat'
	
	def logistic(self,inX):    
		return 1/(1+exp(-inX))
	def gradient_logistic(self,net):
		return multiply(net,1.0-net)
	
	#主函数
	
	def forward_propagate(self):	#前向传播过程
		self.hi=self.dataMat*self.hi_wb.T  #(m*nNode)
		self.ho=self.logistic(self.hi)               #(m*nNode)
		self.ho_bias=concatenate((ones((self.example,1)),self.ho),axis=1) #(m*nNode+1)
		self.yi=self.ho_bias*self.oh_wb.T  #(m*nOut)
		self.yo=self.logistic(self.yi)     #(m*nOut)
	
	def cost(self,err):
		return sum(power(err,2))*0.5
		
	def backward_propagate(self):		#反向传播过程
		#输出层
		Delta=multiply(self.classLabels-self.yo,self.gradient_logistic(self.yo)) #(m*nOut)
		gradient_oh=Delta.T*self.ho_bias  #(nOut*nNode+1)
		#隐含层
		delta=multiply(Delta*self.oh_wb[:,1:],self.gradient_logistic(self.ho)) #(m*nNode)
		gradient_hi=delta.T*self.dataMat  #(nNode*nIn+1)
		return gradient_hi,gradient_oh
		
	def bpnet(self):
		self.normalize1()
		self.normalize2()
		self.init_hiddenWB()  #初始化输入层与隐含层之间的权重向量
		self.init_OutputWB()  #初始化隐含层与输出层之间的权重向量
		dhi=0.0;doh=0.0;  #t-1时刻的权重微分，初始为0
		for i in range(self.iterator_max):
				self.iterator=i+1
				#print 'iter:',i+1
				self.forward_propagate()  #前向传播过程
				err=self.classLabels-self.yo
				sse=self.cost(err)
				self.errorList.append(sse)
				if sse<=self.eb:
					break;
				
				gradient_hi,gradient_oh=self.backward_propagate()  #权重微分
				if i<10000:
					self.oh_wb=self.oh_wb+self.learningRate*gradient_oh   #更新权重
					self.hi_wb=self.hi_wb+self.learningRate*gradient_hi
				else:
					self.hi_wb=self.hi_wb+(1.0-self.mc)*self.learningRate*gradient_hi+self.mc*dhi   #网络调优的做法
					self.oh_wb=self.oh_wb+(1.0-self.mc)*self.learningRate*gradient_oh+self.mc*doh
					dhi=gradient_hi
					doh=gradient_oh
				#self.learningRate=self.learningRate_max-i*(self.learningRate_max-self.learningRate_min)/self.iterator_max   #变学习率
		return self.hi_wb,self.oh_wb
	
	def Desicion_boundary(self,step=100):
		n=self.dataMat.shape[1]
		if n==3:
			amin=min(min(self.dataMat.tolist()))
			amax=max(max(self.dataMat.tolist()))
			print amin,amax
			x=linspace(int(round(amin)),int(round(amax)),step)
			#print x
			#print 'x:',len(x)
			xx=mat(ones((step,step)))
			xx[:,0:step]=x
			yy=xx.T
			xi=[]
			z=ones((len(xx),len(yy)))
			for i in range(len(xx)):
				for j in range(len(xx)):
					xi=mat([1,xx[i,j],yy[i,j]])
					hi_input=self.hi_wb*xi.T
					hi_out=self.logistic(hi_input)
					hi_out_bias=concatenate((mat([1]),hi_out),axis=0)
					#print hi_out_bias.shape
					oh_input=self.oh_wb*hi_out_bias
					oh_out=self.logistic(oh_input)
					z[i,j]=oh_out
					
		return x,z
	
	def classfierLine(self,x,z):
		#print 'x,z:',x,z
		x,y=meshgrid(x,x)
		#print z.shape
		plt.contour(x,y,z,1,colors='black')
	
	def TrendLine(self):
		x=range(0,self.iterator)
		y=log2(self.errorList)
		print y
		plt.plot(x,y,color='r')
		plt.show()
	
		
						
			
a=BPnet()
a.loadDataMat(r'D:\python_learning_dataes\testSet2.txt')
print a.dataMat.shape   #(307,3)
print a.classLabels.shape	#(307,1)
#print a.dataMat
a.normalize1()
a.normalize2()
a.bpnet()
a.drawClassScatter()
x,z=a.Desicion_boundary()
a.classfierLine(x,z)
plt.show()
a.TrendLine()
plt.show()
		