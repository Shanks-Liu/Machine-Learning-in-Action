from imp import reload
from numpy import *

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'pronlems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]#1为侮辱类标签
    return postingList, classVec

#返回不重复的词汇列表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

#返回一个列表，对应词汇列表位置，有的地方是1
def setOfWord2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('The word {} is not in my vocabulary'.format(word))
    return returnVec
 
'''
输入是对list的每行都调用returnVec返回列表堆起来构成的矩阵（在命令行实现），和list每行的类别标签
输出是P(W|C0), P(W|C1), P(C1)
'''
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    #pAbusive = sum(trainCategory) / float(numTrainDocs)
    #为了作业修改开始
    trainCategoryCopy = [number for number in trainCategory if number == 1]
    pAbusive = len(trainCategoryCopy) / float(numTrainDocs)
    #为了作业修改结束
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return -1#为了作业修改为-1，原来是0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for post in listOPosts:
        trainMat.append(setOfWord2Vec(myVocabList, post))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ["love", 'my', 'dalmation']
    thisDoc = array(setOfWord2Vec(myVocabList, testEntry))
    classifyLabel = classifyNB(thisDoc, p0V, p1V, pAb)
    print('{} classified as: {}'.format("".join(testEntry), classifyLabel))

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

#第二节作业二：由表中训练数据学习一个朴素贝叶斯分类器并确定X=(2,"S")的类标签
def assignment():
    list = [[1, 'S'],
            [1, "L"],
            [1, "M"],
            [1, "M"],
            [1, 'S'],
            [2, "L"],
            [2, "S"],
            [2, "S"],
            [2, "L"],
            [2, "L"],
            [2, "M"],
            [3, 'M'],
            [3, "L"],
            [3, "S"],
            [3, "M"],
            [3, "M"]]
    lables = [-1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,-1,1]
    myVocabList = createVocabList(list)
    trainMatrix = []
    for line in list:
        trainMatrix.append(bagOfWords2VecMN(myVocabList, line))
    v0,v1,vA = trainNB0(array(trainMatrix), array(lables))
    test = [2, "S"]
    thisDoc = array(bagOfWords2VecMN(myVocabList, test))
    classifyResult = classifyNB(thisDoc, v0, v1, vA)
    print("this {} is classified as: {}".format(test, classifyResult))

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)#将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
    docList=[]
    classList=[]
    fullText=[]
    for i in range(1, 26):
        wordList = textParse(open(r'email/spam/{}.txt'.format(i)).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open(r'email/ham/{}.txt'.format(i)).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = range(50)#trainingSet全是数
    testSet = []
    #取10个作为测试样本
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p1, p0, pA = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(wordVector, p1, p0, pA) != classList[docIndex]:
            errorCount += 1
    print("错误率为：{}".format(errorCount))

