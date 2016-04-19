# -*- coding: utf-8 -*-
import operator
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from os import  listdir #用于显示文件夹下的文件


def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    temp = tile(inX, (dataSetSize, 1))
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqdiffMat = diffMat ** 2
    sqDistances = sqdiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayoLines = fr.readlines()  # 读取每一行数据，结果应该是list 40920	8.326976	0.953952    3
    numberofLines = len(arrayoLines)
    returnMat = zeros((numberofLines, 3))  # 1000行　３列
    classLabelVector = []
    index = 0
    for line in arrayoLines:
        line = line.strip()  # 去除\r\n
        listFromLine = line.split('\t')  # 得到一个ｌｉｓｔ　储存这一行的４个数据
        returnMat[index, :] = listFromLine[0: 3]  # 拿一到三个数据到ｉｎｄｅｘ行
        classLabelVector.append(int(listFromLine[-1]))  # [-1]是取最后一个数据
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros((shape(dataSet)))  # 一千行，三列，全零元素
    m = dataSet.shape[0]  # ｍ　＝　１０００
    tmp = tile(minVals, (m, 1))  # minVal是个３元素行矩阵，ｔｉｌｅ函数将该矩阵扩展成３列１０００行的矩阵，每行都是ｍｉｎＶａｌ
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))  # 在ｎｕｍｐｙ中／表示矩阵中对应元素相除，linalg.solve(A,B)表示矩阵除法
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.05
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classfierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "the classifier came back with :%d , the real answer is :%d " % (classfierResult, datingLabels[i])
        if (classfierResult != datingLabels[i]): errorCount += 1.0
    print "total error is %f" % (errorCount / float(numTestVecs))

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range (32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m , 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for j in range(mTest):
        fileNameStr = testFileList[j]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' %fileNameStr)
        classfyResult = classify(vectorUnderTest, trainingMat , hwLabels ,3)
        print "the classfier comes back with %d ,the right answer is %d" %(classfyResult, classNumStr)
        if(classfyResult != classNumStr):
            errorCount += 1.0
    print "\nthe total error is %d" %errorCount
    print "\nthe error rate is %f" %(errorCount/float(mTest))





if __name__ == '__main__':
    handwritingClassTest()
