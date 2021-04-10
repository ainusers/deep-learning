import mnist_loader
import network


# 训练集、验证集、测试集
training_data,validation_data,test_data = mnist_loader.load_data_wapper()
print(type(training_data))


# 调用SGD随机梯度下降算法
network = network.Network([784,30,10])
network.SGD(training_data,30,20,3.0,test_data=test_data)