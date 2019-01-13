from imp import reload
from numpy import *


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
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords)  #防止下溢出
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
        return 0


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

#返回长度>2的词汇列表
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
    trainingSet = list(range(50))  #trainingSet全是数
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
    print("错误率为：{}".format(errorCount / len(testSet)))

##########################################
###########作业三与作业四分界线#############
##########################################

def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)#统计次数
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
    import feedparser
    docList=[]
    classList=[]
    fullText=[]
    minLen=min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2 * minLen))
    testSet=[]
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(randIndex)
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0, p1, pA = trainNB0(array(trainMat), array(classList))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(wordVector, p0, p1, pA) != classList[docIndex]:
            errorCount += 1
    print('误差率为: {}'.format(errorCount / len(testSet)))
    return vocabList, p0, p1 


def getTopWords(ny, sf):
    import operator
    vocabList, p0, p1 = localWords(ny, sf)
    topNY=[]
    topSF=[]
    for i in range(len(p0)):
        if p0[i] > -6.0:
            topSF.append((vocabList[i], p0[i]))
        if p1[i] > -6.0:
            topNY.append((vocabList[i], p1[i]))
    sortedSF = sorted(topSF, key=itemgetter(1), revers=True)
    print("*SF*" * 10)
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=itemgetter(1), revers=True)
    print("*NY*" * 10)
    for item in sortedNY:
        print(item[0])
