from numpy import *
import operator
from os import listdir

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    #dataSetSize = dataSet.shape[0]
    #diffMat = tile(inX, (dataSetSize,1)) - dataSet
    #sqDiffMat = diffMat ** 2
    #sqDistances = sqDiffMat.sum(axis=1)
    #distances = sqDistances ** 0.5
    labels = array(labels)
    distances = sqrt(sum(power((inX - dataSet), 2),axis=1))

    sortedDistIndicies = distances.argsort()
    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename, 'r', encoding='utf-8')
    arrayOLines = fr.readlines()
    #arrayOLines[0]=arrayOLines[0].lstrip('\ufeff')
    numberOFLines = len(arrayOLines)
    returnMat = zeros((numberOFLines, 3))
    classLabelVector = []
    index = 0
    
    for line in arrayOLines:
        line.strip()
        listFromLine = line.split()
        returnMat[index, :] = listFromLine[0:3]
        
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        if listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        if listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector

def autoNorm1(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet / tile(ranges, (m,1))
    return normDataSet, ranges, minVals

def autoNorm2(dataSet):
    #标注轴取0，结果正确；如果不标注轴取0，会得到很奇怪的结果，标注轴取1的话，又会报错
    normDataSet = (dataSet - dataSet.min(axis=0)) / (dataSet.max(axis=0) - dataSet.min(axis=0))
    return normDataSet

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm1(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    print(numTestVecs)
    errorcount = 0

    for i in range(numTestVecs):
       # print('>>>>>>>>>>>>>', normMat[i, :].shape, "\n>>>>>>>>>>>>>>", \
                #normMat[numTestVecs:m, :].shape, "\n>>>>>>>>>>>>>>>>>>", \
                #len(datingLabels[numTestVecs:m]))
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], \
                            datingLabels[numTestVecs:m], k=4)
        print("分类结果：{}, 真实结果：{}".format(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorcount += 1
    print("错误率为： {}".format(errorcount/numTestVecs))

def classifyPerson():
    resultList = ["讨厌", '有些喜欢', '非常喜欢']
    percenTats = float(input('玩视频游戏所占时间百分比：')   )
    ffMiles = float(input('每周获得的飞行常客旅程数： '))
    iceCream = float(input('每周消耗的冰淇淋公升数： '))

    filename = 'datingTestSet.txt'
    datingDataMat, datingLabels = file2matrix(filename)
    normMat, ranges, minVals = autoNorm1(datingDataMat)
    inArr = array([percenTats, ffMiles, iceCream]) 
    inArr = autoNorm2(inArr)
    classifierResult = classify0(inArr, normMat, datingLabels, k=3)
    print('这个人或许你{}'. format(resultList[classifierResult-1]))

def img2vector(filename):
    returnVect = zeros((1, 1024))
    with open(filename) as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels=[]
    #返回目录下文件名
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))

    for i in range(m):
        filenameStr = trainingFileList[i]
        #这个文件的真实分类
        classNumber = int(filenameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i, :] = img2vector(r'trainingDigits\{}'.format(filenameStr))

    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)

    for i in range(mTest):
        filenameStr = testFileList[i]
        #这里如果没有加int，下面那个if函数相当于比较的是数值和字符串，就全都统计成错误了
        classNumber = int(filenameStr.split('_')[0])
        testMat = img2vector(r'testDigits/{}'.format(filenameStr))
        testResult = classify0(testMat, trainingMat, hwLabels, 3)
        if testResult != classNumber:
            errorCount += 1.0
    print("错误率为： {}".format(errorCount/mTest))
    




    