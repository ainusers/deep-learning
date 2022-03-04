import numpy as np


# Function: K Means
# -------------
# K-Means is an algorithm that takes in a dataset and a constant
# k and returns k centroids (which define clusters of data in the
# dataset which are similar to one another).
# 数据集需要加一列
# 参数：数据集，分为几类，迭代次数
def kmeans(X, k, maxIt):
    # 返回行列维度
    numPoints, numDim = X.shape
    # 增加一列作为分类标记
    dataSet = np.zeros((numPoints, numDim + 1))
    # 所有行，除了最后一列
    dataSet[:, :-1] = X

    # 随机选取K行，包含所有列
    centroids = dataSet[np.random.randint(numPoints, size=k), :]
    centroids = dataSet[0:2, :]
    # 给中心点分类进行初始化
    centroids[:, -1] = range(1, k + 1)
    # Initialize book keeping vars.
    iterations = 0
    oldCentroids = None

    # Run the main k-means algorithm
    while not shouldStop(oldCentroids, centroids, iterations, maxIt):
        print ("iteration: \n", iterations)
        print ("dataSet: \n", dataSet)
        print ("centroids: \n", centroids)
        # Save old centroids for convergence test. Book keeping.
        oldCentroids = np.copy(centroids)
        iterations += 1
        #根据数据集以及中心点对数据集的点进行归类
        updateLabels(dataSet, centroids)
        # 更新中心点
        centroids = getCentroids(dataSet, k)
    # We can get the labels too by calling getLabels(dataSet, centroids)
    return dataSet


# Function: Should Stop
# -------------
# Returns True or False if k-means is done. K-means terminates either
# because it has run a maximum number of iterations OR the centroids
# stop changing.
# 实现函数循环结束的判断
# 当循环次数达到最大值，或者中心点不变化就停止
def shouldStop(oldCentroids, centroids, iterations, maxIt):
    if iterations > maxIt:
        return True
    return np.array_equal(oldCentroids, centroids)


# Function: Get Labels
# -------------
# Update a label for each piece of data in the dataset.
# 根据数据集以及中心点对数据集的点进行归类
def updateLabels(dataSet, centroids):
    # For each element in the dataset, chose the closest centroid.
    # Make that centroid the element's label.
    # 返回行（点数），列
    numPoints, numDim = dataSet.shape
    for i in range(0, numPoints):
        # 对每一行最后一列进行归类
        dataSet[i, -1] = getLabelFromClosestCentroid(dataSet[i, :-1], centroids)

# 对比一行到每个中心点的距离，返回距离最短的中心点的label
def getLabelFromClosestCentroid(dataSetRow, centroids):
    # 初始化label为中心点第一点的label
    label = centroids[0, -1];
    # 初始化最小值为当前行到中心点第一点的距离值
    # np.linalg.norm计算两个向量的距离
    minDist = np.linalg.norm(dataSetRow - centroids[0, :-1])
    # 对中心点的每个点开始循环
    for i in range(1, centroids.shape[0]):
        dist = np.linalg.norm(dataSetRow - centroids[i, :-1])
        if dist < minDist:
            minDist = dist
            label = centroids[i, -1]
    print ("minDist:", minDist)
    return label


# Function: Get Centroids
# -------------
# Returns k random centroids, each of dimension n.
# 更新中心点
# 参数：数据集（包含标签），k个分类
def getCentroids(dataSet, k):
    # Each centroid is the geometric mean of the points that
    # have that centroid's label. Important: If a centroid is empty (no points have
    # that centroid's label) you should randomly re-initialize it.
    # 初始化新的中心点矩阵
    result = np.zeros((k, dataSet.shape[1]))
    for i in range(1, k + 1):
        # 找出最后一列类别为i 的行集,即求一个类别里面的所有点
        oneCluster = dataSet[dataSet[:, -1] == i, :-1]
        # axis = 0 对行求均值，并赋值
        result[i - 1, :-1] = np.mean(oneCluster, axis=0)
        result[i - 1, -1] = i

    return result


x1 = np.array([1, 1])
x2 = np.array([2, 1])
x3 = np.array([4, 3])
x4 = np.array([5, 4])
# 将点排列成矩阵
testX = np.vstack((x1, x2, x3, x4))

result = kmeans(testX, 2, 10)
print ("final result:")
print (result)
