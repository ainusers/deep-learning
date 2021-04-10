import mnist_loader
import network2 as network


# 训练集、验证集、测试集
training_data,validation_data,test_data = mnist_loader.load_data_wapper()
print(type(training_data))


# 调用SGD随机梯度下降算法 (使用cross-entropy函数)
network = network.Network([784,30,10])
cost = network.CrossEntropyCost()
network.large_weight_initializer()
# monitor_evaluation_accuracy 可以查看学习曲线
network.SGD(training_data,30,20,3.0,test_data=test_data,monitor_evaluation_accuracy = True)