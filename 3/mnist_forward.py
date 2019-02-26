# encoding:utf-8

'''搭建神经网络，在mnist数据集上训练模型，输出手写数字识别准确率。
   mnist数据集：包含7万张黑底白字手写数字图片，其中55000 张为训练集，5000 张为验证集，10000张为测试集。
   每张图片大小为28*28像素，图片中纯黑色像素值为0，纯白色像素值为1。

'''

import tensorflow as tf

# 神经网络节点数784-500-10
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

# 权重，加入正则化，对权重正则化
def get_weight(shape, regularizer):
	w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
	if regularizer != None:
		tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

# 偏置
def get_bias(shape):
	b = tf.Variable(tf.zeros(shape))
	return b

# 前向传播
def forward(x,regularizer):
	w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
	b1 = get_bias([LAYER1_NODE])
	y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

	w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
	b2 = get_bias([OUTPUT_NODE])
	y = tf.matmul(y1, w2) + b2

	return y