import csv
import random
import math
import operator
 
 
def loadDataset(filename, split, trainingSet = [], testSet = []):
    '''
    导入数据
    :param filename:
    :param split: 将数据总集以split为界限 分成训练集和测试集
    :param trainingSet:
    :param testSet:
    :return:
    '''
    with open(filename, 'rt') as csvfile:       # 以逗号为分隔符
        lines = csv.reader(csvfile)             # 读取所有行
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
 
 
def euclideanDistance(instance1, instance2, length):
    '''
    计算euclideanDistance
    :param instance1:
    :param instance2:
    :param length: 维度
    :return:
    '''
    distance = 0
    for x in range(length):
        distance += pow((instance1[x]-instance2[x]), 2)
    return math.sqrt(distance)
 
 
def getNeighbors(trainingSet, testInstance, k):
    '''
    返回最近的k个邻居
    :param trainingSet: 训练集
    :param testInstance: 一个测试实例
    :param k: 参数k
    :return:
    '''
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        #testinstance
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
        #distances.append(dist)
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
        return neighbors
 
 
def getResponse(neighbors):
    '''
    以距离排序，返回最近的几个点
    :param neighbors:
    :return:
    '''
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)  # python3 里的.items()返回的是列表，.iteritems()返回的是一个迭代器
    return sortedVotes[0][0]
 
 
def getAccuracy(testSet, predictions):
    '''
    预测值和实际值的准确率
    :param testSet:
    :param predictions:
    :return:
    '''
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet)))*100.0
 
 
def main():
    #prepare data
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset(r'data.txt', split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    #generate predictions
    predictions = []
    k = 3
    for x in range(len(testSet)):
        # trainingsettrainingSet[x]
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print ('>predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy)+ '%')
 

if __name__ == '__main__':
    main()
