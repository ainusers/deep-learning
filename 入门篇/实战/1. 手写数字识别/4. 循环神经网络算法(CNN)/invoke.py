import mnist_loader
import network3



expanded_training_data, _, _ = network3.load_data_shared(
        "../data/mnist_expanded.pkl.gz")


net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2),
                      activation_fn=ReLU),
        FullyConnectedLayer(
            n_in=40*4*4, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
        FullyConnectedLayer(
            n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
        SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)],
        mini_batch_size)


net.SGD(expanded_training_data, 40, mini_batch_size, 0.03,
            validation_data, test_data)


# 调用SGD随机梯度下降算法 (使用cross-entropy函数)
network = network.Network([784,30,10])
cost = network.CrossEntropyCost()
# 这里去掉这种方式，使用默认的default_weight_initializer(正态分布算法初始化参数)
# network.large_weight_initializer()
network.SGD(training_data,30,10,0.5,5.0,evaluation_data = validation_data,
            monitor_evaluation_accuracy=True,monitor_evaluation_cost=True,
            monitor_traning_accuracy=True,monitor_traning_cost=True)




















