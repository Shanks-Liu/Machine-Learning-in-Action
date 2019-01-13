from numpy import *
from imp import reload

#载入数据
def loadDataSet():
    dataMat=[]
    labelMat=[]
    with open("testSet.txt") as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
        return dataMat, labelMat #shape为（100,3） （100，）

#sigmoid函数
def sigmoid(inX):
    return 1.0 / (1+exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose() #并没有把两个输入本身转化成矩阵
    print(type(labelMat))
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        #推导出来的对w的偏导数，此时是矩阵乘法
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights #X.getA()可以把X变成数组，weight的shape为（3，1），此时weights的类型是矩阵

#画图时注意两个数组的形状，只能是(60,)这种，如果是(1,60)也不行
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)#???
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    print(">>>>>>>>>>>>>>", x, x.shape)
    y = (-weights[0][0] - weights[1][0] * x) / weights[2][0]
    print(">>>>>>>>>>>>>>>>Y: ", y, y.shape)
    #x.shape = y.shape#属性也是可以改的
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel("X2")
    plt.show()

#每次用一个样本更新参数，称为随机梯度算法
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

#创建一个临时变量，用来随机抽取后删除索引
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    print(">>>>>>>>>>>type(dataMatrix): ", type(dataMatrix), len(dataMatrix))
    dataMatrix = array(dataMatrix)#如果不把这个列表转化成数组的话，后面的乘法会报错
    m, n = shape(dataMatrix)
    weights = ones(n)
    print(">>>>>>>>>weights: ", weights, weights.shape)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0+j+i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))#此时是数组相乘
            #print(">>>>>>>>>>>type(weights): ", type(weights))
            #print(">>>>>>>>>>>type(dataMatrix[randIndex]): {} \n,\
                   #>>>>>>>>>>>dataMatrix[randIndex]: {}".format(type(dataMatrix[randIndex]), dataMatrix[randIndex]))
            error = classLabels[randIndex] - h
            #print(">>>>>>>>>>>h: ", h)
            #此时是数相乘，如果前面没有把dataMatrix转化成数组的话，列表乘以一个实数是把列表的长度变长的
            weights += alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])#如果是range的话，是不能删掉索引的
        if j%50 == 0:
            print(">>>>>>>>>第{}次迭代， weight为： {}, 误差为: {}".format(j, weights, error))
    return weights

#预测
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(trainingSet, trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
#有次出bug是因为上面这行忘记把字符串变成浮点数了，我以为下面输入的时候把它变成array，就自动全变成数了，其实不是
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = float(errorCount) / numTestVec
    print("The Errorrate is: {}".format(errorRate))
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
        averageErrorRate = errorSum / float(numTests)
    print("经过{}次迭代， 平均误差率为{}".format(numTests, averageErrorRate))
        

 