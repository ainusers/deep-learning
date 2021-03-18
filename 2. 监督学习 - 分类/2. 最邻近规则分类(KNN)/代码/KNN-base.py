from sklearn import neighbors
from sklearn import datasets


# 实例化knn对象
knn = neighbors.KNeighborsClassifier()

# 导入数据集
iris = datasets.load_iris()
print (iris)

# 建立knn模型
knn.fit(iris.data, iris.target)

# 测试集验证模型
predictedLabel = knn.predict([[0.1, 0.2, 0.3, 0.4]])
print (predictedLabel)