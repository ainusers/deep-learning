import chainer
from chainer import configuration
from chalner.dataset import convert
from chainer.iterators import MultiprocessIterator,SerialIterator
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer import serializers


# Network definition (全连接层神经网络定义)
class MLP(chainer.Chain):
    def __init__(self,n_units,n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None,n_units)  # n_in -> n_units
            self.l2 = L.Linear(None,n_units)  # n_units -> n_units
            self.l3 = L.Linear(None,n_out)  # n_out -> n_units

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

# 参数定义
batchsize = 100  # 批次大小
epochs = 20  # 对训练集重复训练多少次
gpuid = 1
outdir = 'result'
unit = 1000  # 隐藏层神经元数量

print(f'# GPU:  {gpuid}')
print(f'# unit:  {unit}')
print(f'# Minibatch-size:  {batchsize}')
print(f'# epochs:  {epochs}')
print('')


# set up a neural network to train (构建一个神经网络模型到第一个GPU上)
model = L.Classifier(MLP(unit,10))
if gpuid >= 0:
    # use GPU
    chainer.backends.cuda.get_device_from_id(gpuid).use()
    # copy the model to the gpu
    mode.to_gpu()


# set an optimizer (设置优化算法)
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)


# load the dataset (加载数据集[训练集和测试集])
train,test = chainer.datasets.get_fashion_mnist()
train_count,test_count = len(train),len(test)

train_iter = SerialIterator(train,batchsize)
# repeat=False表示不会无限循环采样训练集
# shuffle=False表示不会随机打乱样本顺序
test_iter = SerialIterator(test,batchsize,repeat=False,shuffle=False)

sum_accuracy = 0
sum_loss = 0


# 模型训练(当小于设定的epoch次数会一直训练)
while train_iter.epoch < epochs:
    batch = brain_iter.next()  # 取样本
    x_array,t_array = convert.concat_examples(batch,gpuid)
    x = chainer.Variable(x_array)
    t = chainer.Variable(t_array)
    optimizer.update(mode,x,t) # 更新网络参数
    # 损失函数和准确度累加计算
    sum_loss += float(model.loss.data) * len(t.data)
    sum_accuracy += float(model.accuracy.data) * len(t.data)

    # 如果完成一轮epoch训练输出每轮的损失函数和准确度
    if train_iter.is_new_epoch:
        print(f'epoch:  {train_iter.epoch}')
        print(f'train mean loss:  {sum_loss / train_count}, accuracy: {sum_accuracy / train_count}')

        # evaluation (每轮学习之后，进行一次测试数据集的测试)
        sum_accuracy = 0
        sum_loss = 0
        # enable evaluation mode
        with configuration.using_config('train',False):
            # this is optional but can reduce computational
            with chainer.useing_config('enable_backprop',False):
                for batch in test_iter:
                    x,t = conver.concat_examples(batch,gpuid)
                    x = chainer.Variable(x)
                    t = chainer.Variable(t)
                    loss = model(x,t)
                    sum_loss += float(loss.data) * len(t.data)
                    sum_accuracy += (float(model.accuracy.data) * len(t.data))
        # 迭代器状态还原
        test_iter.reset()
        print(f'test mean loss: {sum_loss / test_count}, accuracy: {sum_accuracy / test_count}')
        sum_accuracy = 0
        sum_loss = 0


# save the model and the optimizer (保存模型和优化参数)
print('save the model')
serializers.save_npz('{}/mlp.model'.format(outdir),model)

# 加载模型
# serializers.load_npz('{}/mlp.model'.format(outdir),model)