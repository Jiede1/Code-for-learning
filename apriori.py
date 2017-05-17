#coding:utf-8
def loadDataSet():
	return [[1,3,4],[2,3,4],[1,2,3,5],[2,5]]
# 构建集合C1
def createC1(dataSet):

    # 创建一个空集合 C1
    C1 = []

    # 遍历数据集
    for transaction in dataSet:
        # 遍历记录中的每一项
        for item in transaction:
            # 如果没有在C1中，则添加到C1
            if not [item] in C1:
                C1.append([item])

    # 对 C1 列表进行排序
    C1.sort()

    # 将 C1 中每个单元素列表映射到 frozenset()
    return map(frozenset, C1)

# 由Ck生成Lk，参数分别为：
# 1. 数据集
# 2. Ck 候选项集
# 3. 最小支持度
def scanD(D, Ck, minSupport):
    # 创建空字典
    ssCnt = {}

    # 遍历数据集中所有记录
    for tid in D:
        # 遍历Ck中的所有候选集
        for can in Ck:
            # 如果候选集是记录的一部分
            if can.issubset(tid):
                # 则对 ssCnt 字典中的候选集计数加一
                if not ssCnt.has_key(can): ssCnt[can]=1
                else: ssCnt[can] += 1

    # 开始计算支持度

    numItems = float(len(D))

    # 包含满足最小支持度的集合列表
    retList = []

    # 支持度字典
    supportData = {}

    # 遍历集合频次字典
    for key in ssCnt:
        # 计算支持度
        support = ssCnt[key]/numItems

        # 如果大于最小支持度则存入返回列表
        if support >= minSupport:
            retList.insert(0,key)

        # 保存支持度
        supportData[key] = support
    return retList, supportData

# 创建候选项集 Ck
# 输入参数为频繁项集列表 Lk 及项集元素个数 k
def aprioriGen(Lk, k):
    # 创建空列表并计算Lk中的元素数量
    retList = []
    lenLk = len(Lk)

    # 两层for循环来比较Lk中每个元素与其他元素
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            # 取列表中的两个集合进行比较
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            # 如果集合前面k-2个元素都相同则将集合合成大小为k的新集合
            if L1==L2:
                retList.append(Lk[i] | Lk[j])
    return retList

# Apriori 算法实现，输入参数为数据集和支持度
def apriori(dataSet, minSupport = 0.5):
    # 创建C1
    C1 = createC1(dataSet)

    # 初始化 D，L1，及k值
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    # 将 L1 存入 L列表，L 列表包含 L1,L2...等。
    L = [L1]
    k = 2

    # 循环遍历L列表，直到L[k-2]为空
    while (len(L[k-2]) > 0):
        # 由 L[k-2] 创建候选项集列表 Ck
        Ck = aprioriGen(L[k-2], k)

        # 由Ck创建Lk
        Lk, supK = scanD(D, Ck, minSupport)

        # 更新支持度字典
        supportData.update(supK)

        # 将 Lk 加入到 L 列表
        L.append(Lk)
        k += 1

    # 返回 L 列表及支持度字典
    return L, supportData

# 计算可信度值，返回一个满足最小可信度要求的规则列表
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    # 创建返回列表
    prunedH = []

    # 遍历频繁项集H
    for conseq in H:
        print "conseq:",conseq
        # 计算可信度值
        conf = supportData[freqSet]/supportData[freqSet-conseq] # 如果大于最小可信度，则输出规则信息
        if conf >= minConf:
            print freqSet-conseq,'-->',conseq,'conf:',conf
            # 保存规则及频繁项集
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

# 由频繁项集生成关联规则
# 输入参数为频繁项集和列表H，支持度字典，规则集合，最小可信度
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    # H中频繁集大小
    m = len(H[0])
    print "m:",m
    # 判断频繁项集是否可以移除m大小的子集
    if (len(freqSet) > (m + 1)):
        # 创建新的候选项集 Ck
        Hmp1 = aprioriGen(H, m+1)
        print "hmp1:",Hmp1
        # 计算可信度并返回频繁项集列表
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        print "back htm:",Hmp1
        # 如果返回的集合大于1则继续递归进一步组合规则
        if (len(Hmp1) > 1):
            print "Yes"
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

# 生成规则列表
# 参数：频繁项集列表，支持度字典，最小可信度阈值
def generateRules(L, supportData, minConf=0.7):
    print "L:",L
	# 初始化返回的规则列表
    bigRuleList = []

    # 遍历频繁项集列表
    for i in range(1, len(L)):
        # 遍历每个频繁项集
        for freqSet in L[i]:
            print "freqSet:",freqSet
            # 构建只包含单个元素集合的列表 H1 
            H1 = [frozenset([item]) for item in freqSet]
            print "H1:",H1
            # 如果频繁集的元素数超过2则进一步的合并
            # 否则对有两个元素的项集计算可信度
            if (i > 1):
                print i
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                print i
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)

    # 返回满足可信度要求的规则列表
    return bigRuleList

dataSet=loadDataSet()
print "dataSet:",dataSet
D=map(set,dataSet)
print "d:",D
L, suppData = apriori(dataSet, minSupport=0.2)

rules = generateRules(L, suppData, minConf=0.50)
print rules