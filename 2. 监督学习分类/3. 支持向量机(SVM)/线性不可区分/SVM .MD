# 支持向量机算法


# 解决线性不可分情况
![Image text](SVM_解决线性不可分情况.jpg)
1. 利用一个非线性的映射把原数据集中的向量点转化到一个更高维度的空间中
2. 在这个高维度的空间中找一个线性的超平面来根据线性可分的情况处理


# 如何利用非线性映射把原始数据转化到高维中?
算法 ≈ 核函数 (解决计算内积时算法复杂度非常高的问题)


# 常用的核函数(kernel functions)
h度多项式核函数(polynomial kernel of degree h)
高斯径向基核函数(Gaussian radial basis function kernel)
S型核函数(Sigmoid function kernel)

注：在实战中，尝试不同的kernel，根据结果准确度来选择使用哪一种核函数


# SVM扩展可解决多个类别分类问题
SVM只能解决两类问题，但是可以换个角度解决多类问题 (如：迭代版的当前类和其他类)