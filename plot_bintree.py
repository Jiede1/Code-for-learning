#coding:UTF-8
import matplotlib.pyplot as plt  
decisionNode = dict(boxstyle = "sawtooth",fc="0.8")  
leafNode = dict(boxstyle="round4",fc="0.8")  
arrow_args = dict(arrowstyle="<-")  
  
def plotNode(nodeTxt,centerPt,parentPt,nodeType):  
    createPlot.ax.annotate(nodeTxt,xy=parentPt,xycoords = 'axes fraction',
        xytext=centerPt,textcoords='axes fraction',va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)  

def createPlot(): 
    fig = plt.figure(1, facecolor='white')#创建一个新图形 
    fig.clf()#清空绘图区 
    createPlot.ax= plt.subplot(111,frameon=False) 
    #在图中绘制两个代表不同类型的树节点 
    plotNode('a decision node',(0.5,0.1),(0.1,0.5),decisionNode) 
    plotNode('a leaf node',(0.8,0.1),(0.3,0.8),leafNode) 
    plt.show() 
createPlot()