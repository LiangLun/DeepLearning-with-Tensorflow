# encoding:utf-8

import tensorflow as tf

#1.定义神经网络的输入、参数和输出，定义前向传播过程
# regularizer是正则化系数。正则化是在损失函数中加入刻画模型复杂程度的指标，解决过拟合问题。一般都是对权重w进行正则化的
def get_weight(shape, regularizer):
	w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
	# 这里用的是l2正则化。tf.add_to_collection：把变量放入一个集合（这里集合是losses），把很多变量变成一个列表/向量，列表示例为[1.0,2.0,3.0]，方面后面将losses加到老的loss上
	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

def get_bias(shape):
	b = tf.Variable(tf.constant(0.01, shape=shape))
	return b

def forward(x,regularizer):
	w1 = get_weight([2,11], regularizer)
	b1 = get_bias([11])
	# 这里用激活函数relu来实现去线性化
	y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

	w2 = get_weight([11,1], regularizer)
	b2 = get_bias([1])
	y = tf.matmul(y1,w2)+b2  #输出不用激活
	return y
	
