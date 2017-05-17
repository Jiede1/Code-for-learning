#coding:utf-8
# 定义树节点类型
class treeNode:
    # 初始化FP树节点
    # 参数：节点名字，计数值，父节点
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        # 链接相似元素项
        self.nodeLink = None
        self.parent = parentNode

        # 子节点集合
        self.children = {} 

    # 增加计数
    def inc(self, numOccur):
        self.count += numOccur

    # 以文本方式显示树结构，便于调试    
    def disp(self, ind=1):
        print '  '*ind, self.name, ' ', self.count
        for child in self.children.values():
            child.disp(ind+1)
			
# 更新节点 nodeLink 链表，确保节点链接指向树中该元素项的每一个实例
# 遍历nodeToTest的 nodeLink 链表，直到表尾部，然后再链接上 targetNode
def updateHeader(nodeToTest, targetNode):
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

# 更新树结构
# 参数：排序后的频繁项集，树节点，头指针表，增加频次
def updateTree(items, inTree, headerTable, count):
    # 测试集合中的第一个元素是否为树的子节点
    if items[0] in inTree.children:
        # 增加子节点的计数值
        inTree.children[items[0]].inc(count)
    else:
        # 如果不是子节点则创建新的节点并添加到FP树中
        inTree.children[items[0]] = treeNode(items[0], count, inTree)

        # 更新头指针表
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])

    # 对剩下的元素项进行递归调用，继续让 FP 生长新的节点
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

# 构建 FP 树
# 参数：数据集，最小支持度        
def createTree(dataSet, minSup=1):

    headerTable = {}

    # 第一次遍历数据集
    for trans in dataSet:
        # 遍历每条记录里的每个元素项
        for item in trans:
            # 统计出现的频次
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]

    # 删除没有满足最小支持度的项目
    for k in headerTable.keys():
        if headerTable[k] < minSup: 
            del(headerTable[k])

    freqItemSet = set(headerTable.keys())

    # 如果所有项都没有满足最小支持度要求则返回
    if len(freqItemSet) == 0: return None, None

    # 扩展头指针表，包含频次及指向每种类型第一个元素项的 nodeLink 指针
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]

    # 初始化创建FP树
    retTree = treeNode('Null Set', 1, None)

    # 第二次遍历数据集
    for tranSet, count in dataSet.items():
        localD = {}

        # 只考虑先前获得的频繁项
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]

        # 如果当前记录中包含了频繁项
        if len(localD) > 0:
            # 则对localD中的频繁项进行排序，见P227 图12-2
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            print "orderItems:",orderedItems
            # 使用排序后的频繁项集，调用 updateTree 让FP树生长
            updateTree(orderedItems, retTree, headerTable, count)

    # 返回FP树结构头指针表
    return retTree, headerTable
	
def loadSimpDat():
    simpDat=[['r','z','h','j','p'],
	['z','y','x','w','v','u','t','s'],
	['z'],['r','x','n','o','s'],
	['y','r','x','z','q','t','p'],
	['y','z','x','e','q','s','t','m']]
    return simpDat
def createInitSet(dataSet):
    retDict={}
    for trans in dataSet:
        retDict[frozenset(trans)]=1
    return retDict


# 从树的叶子节点向上迭代遍历整棵树，并收集所有元素项的名字
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

# 为给定的元素项查找条件模式基
def findPrefixPath(basePat, treeNode):
    condPats = {}

    # 遍历元素项的 nodeLink 链表
    while treeNode != None:
        prefixPath = []
        # 收集迭代上溯整棵树过程中遇到的节点名称
        ascendTree(treeNode, prefixPath)

        # 将节点名称列表添加到条件模式基字典中
        if len(prefixPath) > 1: 
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink

    # 返回条件模式基字典
    return condPats

# FP 树递归查找频繁项集
# 参数：FP 树，头指针表，最小支持度，
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):

    # 对头指针表元素项按出现频次进行排序，顺序从小到大
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1])]
    print "bigL:",bigL
    # 遍历排序后的集合
    for basePat in bigL:
        print "basePat:",basePat
        # 添加频繁项到频繁项集中
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)

        # 查找条件模式基
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        print "condPattBases:",condPattBases
        # 创建条件FP树
        myCondTree, myHead = createTree(condPattBases, minSup)
        print "myhead:",myHead
        # 判断该树是否只包含一个元素项
        if myHead != None:
            print "yes"
            print "conditional tree for ",newFreqSet
            myCondTree.disp(1)
            # 递归调用 mineTree 创建条件FP树
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
simpDat=loadSimpDat()
initSet=createInitSet(simpDat)
myFPTree,myHeaderTab=createTree(initSet,3)
myFPTree.disp()
freqItems=[]
mineTree(myFPTree,myHeaderTab,1,set([]),freqItems)
print freqItems