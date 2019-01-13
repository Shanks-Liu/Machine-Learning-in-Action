from math import log
import operator
import pickle
'''
ID3算法
'''


#计算香农熵
'''
返回数据集的信息熵
'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    
    for featVec in dataSet:
        currentLabel = featVec[-1]
        #书上步骤繁琐，用下面这个语句一行就行
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    shannonEnt = 0.0

    for key in labelCounts.keys():
        prob = labelCounts[key] / numEntries
        shannonEnt -= prob * log(prob, 2)
    
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,"yes"],
              [1,1,"yes"],
              [1,0,"no"],
              [0,1,"no"],
              [0,1,"no"]]
    labels = ["no surfacing", 'flippers']
    return dataSet, labels

#划分数据集
'''
返回按那个轴的特征符合value的嵌套列表
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

#选择信息增益最高的划分方式，返回的是一个索引值
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / len(dataSet)
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            print("最优划分是第{}个特征，信息收益是{}".format(i, infoGain))
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#找到次数最多的类标签
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#创建树
def createTree(dataSet, labels):
    sublabels = labels[:]
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]#最优特征的名称
    myTree = {bestFeatLabel:{}}
    del (sublabels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels1 = sublabels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),
                                                 subLabels1)
    return myTree

#使用决策树分类
'''
特征名称列表featLabels和testVec的顺序要对应
'''
def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == "dict":
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)

def grabTree(filename):
    with open(filename, 'rb') as fr:
        return pickle.load(fr)