# hierarchical clustering算法


# 算法步骤
假设有N个待聚类的样本，对于层次聚类来说，步骤：
1、（初始化）把每个样本归为一类，计算每两个类之间的距离，也就是样本与样本之间的相似度；
2、寻找各个类之间最近的两个类，把他们归为一类（这样类的总数就少了一个）；
3、重新计算新生成的这个类与各个旧类之间的相似度；
4、重复2和3直到所有样本点都归为一类，结束


# 如何判断两个类之间的相似度
SingleLinkage：又叫做 nearest-neighbor ，就是取两个类中距离最近的两个样本的距离作为这两个集合的距离，也就是说，最近两个样本之间的距离越小，这两个类之间的相似度就越大。容易造成一种叫做 Chaining 的效果，两个 cluster 明明从“大局”上离得比较远，但是由于其中个别的点距离比较近就被合并了，并且这样合并之后 Chaining 效应会进一步扩大，最后会得到比较松散的 cluster 。
CompleteLinkage：这个则完全是 Single Linkage 的反面极端，取两个集合中距离最远的两个点的距离作为两个集合的距离。其效果也是刚好相反的，限制非常大，两个 cluster 即使已经很接近了，但是只要有不配合的点存在，就顽固到底，老死不相合并，也是不太好的办法。这两种相似度的定义方法的共同问题就是指考虑了某个有特点的数据，而没有考虑类内数据的整体特点。
Average-linkage：这种方法就是把两个集合中的点两两的距离全部放在一起求一个平均值，相对也能得到合适一点的结果。
average-linkage的一个变种就是取两两距离的中值，与取均值相比更加能够解除个别偏离样本对结果的干扰。


# 算法思想
通过计算实例(每行测试数据)相似度，通过相似度算法进行两两节点合并，之后进一步计算合并之后的实例相似度,直到node节点=1停止模型训练


# 算法结构图
![Image text](hierarchical clustering算法结构图.png)