# -*- coding: utf-8 -*-
import operator
from numpy import *
import matplotlib
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    # plt.show()
    datingClassTest()
