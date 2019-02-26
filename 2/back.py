# encoding:utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import generateds
import forward


STEPS = 40000
BATCH_SIZE = 30
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.999
REGULARIZER = 0.01

def back():
	x = tf.placeholder(tf.float32, shape=(None, 2))
	y_ = tf.placeholder(tf.float32, shape=(None, 1))
	X, Y_, Y_c = generateds.generateds()
	y = forward.forward(x, REGULARIZER)

	#-------2.定义损失函数及反向传播方法-------
	# 这里将之前的定长学习率改进成指数衰减型学习率。trainable=False用来声明global_step不是训练变量，一定要要。
	# 指数衰减型学习率的计算公式：learning_rate = learning_rate_base * (learning_rate_decay^(global_step/(总样本数/BATCH_SIZE)))
	# 在训练模型时，使用指数衰减学习率可以使模型在训练的前期快速收敛接近较优解，又可以保证模型在训练后期不会有太大波动。
	global_step = tf.Variable(0, trainable=False)
	# global_step=tf.Variable(0)
	# global_step=tf.Variable(0,trainable=True)
	# tf.train.exponential_decay是tensorflow中
	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,
		global_step,
		300 / BATCH_SIZE, # 总样本数/BATCH_SIZE
		LEARNING_RATE_DECAY,
		staircase=True) # 当staircase = True时，指数global_step/(总样本数/BATCH_SIZE)会取整；当staircase = False时,不取整，是光滑直线

	#定义损失函数，用的MMSE均方误差+正则化。tf.get_collection是从一个集合中取出变量,把很多变量变成一个列表(向量):列表示例为[1.0,2.0,3.0]
	loss_mse = tf.reduce_mean(tf.square(y - y_))
	# tf.add_n：把一个列表的东西都依次加起来，是一个常数
	loss_total = loss_mse + tf.add_n(tf.get_collection('losses')) # 不可以将tf.add_n去掉，因为tf.get_collection后是[]，tf.add_n后变成常数

	#定义反向传播方法：包含正则化
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)

	#-------3.生成对话，训练steps轮-------
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		for i in range(STEPS):
			start = (i * BATCH_SIZE) % 300
			end = start + BATCH_SIZE
			sess.run(train_step, feed_dict={x:X[start:end], y_:Y_[start:end]})
			if i % 2000 == 0:
				loss_v = sess.run(loss_total, feed_dict={x:X, y_:Y_})
				print("After %d steps,loss is:%f" % (i, loss_v))

		xx,yy = np.mgrid[-3:3:0.01, -3:3:0.01]
		grid = np.c_[xx.ravel(), yy.ravel()]
		probs = sess.run(y, feed_dict={x:grid})
		probs = probs.reshape(xx.shape)

	plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
	plt.contour(xx, yy, probs, levels=[.5])
	plt.show()


if __name__ == '__main__':
	back()
