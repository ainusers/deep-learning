import mnist_loader
import network2 as network


# 训练集、验证集、测试集
training_data,validation_data,test_data = mnist_loader.load_data_wapper()
print(type(training_data))


# 调用SGD随机梯度下降算法 (使用cross-entropy函数)
network = network.Network([784,30,10])
cost = network.CrossEntropyCost()
# 这里去掉这种方式，使用默认的default_weight_initializer(正态分布算法初始化参数)
# network.large_weight_initializer()
network.SGD(training_data,30,10,0.5,5.0,evaluation_data = validation_data,
            monitor_evaluation_accuracy=True,monitor_evaluation_cost=True,
            monitor_traning_accuracy=True,monitor_traning_cost=True)




















