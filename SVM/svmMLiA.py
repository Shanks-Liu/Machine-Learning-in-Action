from numpy import *


def loadDataSet(fileName):
    '''
    载入数据
    接收文件夹名称
    返回的是列表
    '''
    dataMat = []
    labelMat = []
    with open(fileName) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split('\t')  #stirp方法和split方法接收的都是字符串,strip返回字符串，split返回列表
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

'''
辅助函数，选一个和输入不同的下标
'''
def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0,m))
    return j


'''
辅助函数，最终确定alpha
'''
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).T #行变成列
    b = 0
    m, n = dataMatrix.shape
    alphas = mat(zeros((m, 1)))
    iter = 0
    while(iter < maxIter):
        alphaParisChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i,:].T)) + b #gi
            Ei = fXi - float(labelMat[i]) #gi - yi
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
                ((labelMat[i] * Ei > toler) and (alphas[i] > 0)): #判断是否违背KKT条件
                j = selectJrand(i, m) #到了第二个alpha
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j,:].T)) + b #gi
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphaJold - alphaIold)
                    H = min(C, C + alphaJold - alphaIold)
                else:
                    L = max(0, alphaIold + alphaJold)
                    H = min(alphaIold + alphaJold, C)
                if L == H:
                    print('L == H')
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T -\
                     dataMatrix[i, :] * dataMatrix[i, :].T -\
                     dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if(abs(alphas[j] - alphaJold) < 0.00001):
                    print("J not moving enough")
                    continue
                alphas[i] += labelMat[i] * labelMat[j] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                    dataMatrix[i, :] * dataMatrix[i, :].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * \
                    dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
                    dataMatrix[i, :] * dataMatrix[j, :].T - \
                    labelMat[j] * (alphas[j] - alphaJold) * \
                    dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i] < C):
                    b = b1
                elif (0 < alphas[j] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaParisChanged += 1
                print(f"iter: {iter}, i: {i}, paris changed: {alphaParisChanged}")
        if (alphaParisChanged == 0):
            iter += 1
        else:
            iter = 0
        print(f"iteratoin number: {iter}")
    return b, alphas

class optStruct:
    '''
    数据结构
    '''
    def __init__(self, dataMatIn, classLabels, C, toler, KTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = dataMatIn.shape[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], KTup)

#计算误差,g(xk)
def calcEk(oS, k):
    '''
    计算误差g(xk),有了核函数后內积运算替换成核函数，称之为"核技巧"
    '''
    #fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k]) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0] 
    if(len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

#更新误差缓存
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    """
	优化的SMO算法，内部循环
	Parameters：
		i - 标号为i的数据的索引值
		oS - 数据结构
	Returns:
		1 - 有任意一对alpha值发生变化
		0 - 没有任意一对alpha值发生变化或变化太小
	"""
    Ei = calcEk(oS, i)
    if((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
        ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei) #选择步长最大的第二个alpha
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        #核技巧
        #eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T 
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print('eta >= 0')
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("J not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        """ b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
                    oS.X[i, :] * oS.X[i, :].T - \
                    oS.labelMat[j] * (oS.alphas[j] - alphaJold) * \
                    oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
                    oS.X[i, :] * oS.X[j, :].T - \
                    oS.labelMat[j] * (oS.alphas[j] - alphaJold) * \
                    oS.X[j, :] * oS.X[j, :].T """
        #核技巧
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] -\
            oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - \
            oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i] < oS.C):
            oS.b = b1
        elif (0 < oS.alphas[j] < oS.C):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, KTup=('lin', 0)):
    '''
    完整版SMO算法,一开始会把输入都变成矩阵
    外部循环
    返回b和alpha
    '''
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, KTup)
    iter = 0
    entireSet = True  
    alphaParisChanged = 0
    while(iter < maxIter) and ((alphaParisChanged > 0) or (entireSet)):
        alphaParisChanged = 0
        if entireSet:  #为真时在整个训练集上遍历
            for i in range(oS.m):
                alphaParisChanged += innerL(i, oS)
                print(f"全样本遍历，第{iter}次迭代，样本{i},alpha优化次数{alphaParisChanged}")
            iter += 1
        else:  #为假时在支持向量上遍历
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaParisChanged += innerL(i, oS)
                print(f"非边界遍历，第{iter}次迭代，样本{i},alpha优化次数{alphaParisChanged}")
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaParisChanged == 0):
            entireSet = True
        print(f"迭代次数{iter}")
    return oS.b, oS.alphas

def calcWs(alphas, dataArr, classLabels):
    '''
    根据alpha计算w,得到超平面
    '''
    X = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n  = X.shape
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w

def kernelTrans(X, A, Ktup):
    '''
    添加一个核函数, A是X其中一行,返回行数和矩阵相同的列向量
    '''
    m, n = X.shape
    K = mat(zeros((m, 1)))
    if Ktup[0] == "lin":
        K = X * A.T
    elif Ktup[0] == "rbf":
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * Ktup[1] ** 2))
    else:
        raise NameError("Houston we have a problem, That Kernel is not recognized")
    return K

def testRbf(k1=1.3):
    '''
    用高斯径向基核函数在训练集和验证集上的误差率
    '''
    dataArr, labelArr = loadDataSet("testSetRBF.txt")
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ("rbf", k1))
    datMat = mat(dataArr)
    labelMat = mat(labelArr).T
    svInd = nonzero(alphas.A > 0)[0]   #取支持向量索引
    sVs = datMat[svInd]  #在矩阵索引中用一个数组索引，返回的是对应的行组成的矩阵
    labelSV = labelMat[svInd]
    print(f'There are {sVs.shape[0]} support vector')
    m, n = datMat.shape
    errorCount = 0.0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))  #返回核函数矩阵
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b  #预测 g(xk)
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print(f'训练误差率为： {errorCount / m}')
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorcount = 0.0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).T
    m, n = datMat.shape
    for i in range(m):
        #用原来数据的sVs和现在数据的每一行来得到核函数
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print("测试集误差率为： {}".format(errorCount / m))

def img2vector(filename):
    '''
    接收文件名，返回行向量
    '''
    returnVect = zeros((1, 1024))
    with open(filename) as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    '''
    接收文件夹名
    把里面的文件挨个读取转化为trainingMat数组, hwLabels列表返回
    '''
    from os import listdir
    hwLabels = []
    trainingfileList = listdir(dirName)
    m = len(trainingfileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingfileList[i]
        fileStr = fileNameStr.split('.')[0]  #这个点是用来把后缀名分开的
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] =  img2vector(dirName + '/' + fileNameStr)
    return trainingMat, hwLabels

def testDigits(KTup=('rbf', 10)):
    dataArr, labelArr = loadImages('trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, KTup)
    datMat = mat(dataArr)
    labelMat = mat(labelArr).T
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print(f"这里有{sVs.shape[0]}个支持向量")
    m, n = datMat.shape
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i], KTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelMat[i]):
            errorCount += 1
    print("训练误差为： {}".format(errorCount / m))
    dataArr, labelArr = loadImages('testDigits')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).T
    m, n = datMat.shape
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], KTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelMat[i]):
            errorCount += 1
    print("测试误差为: {}".format(errorCount / m))

