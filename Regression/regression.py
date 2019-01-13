from numpy import *

def loadDataSet(fileName):
    '''
    读取文件的函数，返回数据二维列表，标签列表
    '''
    featNum = len(open(fileName).readline().strip().split('\t')) - 1
    dataMat = []
    labelMat = []
    with open(fileName) as fr:
        for line in fr.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            for i in range(featNum):
                lineArr.append(curLine[i])
            dataMat.append(lineArr)
            labelMat.append(curLine[-1])
    return dataMat, labelMat
            
def standRegres(xArr, yArr):
    xMat = mat(xArr, dtype=float32)  #因为前面读取的是字符串。如果不加数据类型，会报错ValueError: data type must provide an itemsize
    yMat = mat(yArr, dtype=float32).T
    print('>>>>>>>>>>>xMat: ', repr(xMat))
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:  #判断行列式
        print('该矩阵不可逆')
        return
    ws = xTx.I * xMat.T * yMat
    return ws