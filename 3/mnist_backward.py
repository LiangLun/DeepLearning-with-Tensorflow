# encoding:utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model"

def backward(mnist):
	x = tf.placeholder(tf.float32, [None,mnist_forward.INPUT_NODE])
	y_ = tf.placeholder(tf.float32, [None,mnist_forward.OUTPUT_NODE])
	y = mnist_forward.forward(x, REGULARIZER)
	global_step = tf.Variable(0, trainable=False)

	# 用了交叉熵和softmax
	# 注意ce的两种不同的tensorflow用法，其结果都一样。第一个的labels大小是BATCH_SIZE，第二个的labels大小是BATCH_SIZE*NUM_CLASSES
	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1)) # 只能用于单分类问题
	# ce = tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_) # 可用于多分类
	cem = tf.reduce_mean(ce)
	loss = cem + tf.add_n(tf.get_collection('losses'))

	# 指数衰减学习率
	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,
		global_step,
		mnist.train.num_examples/BATCH_SIZE,
		LEARNING_RATE_DECAY,
		staircase=True)

	# 定义反向传播算法
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
	
	# 滑动平均（影子值）：记录了一段时间内模型所有参数w和b过往各自的平均，增加了模型的范化性
	# 每次运行变量更新时，影子的更新公式为：参数最新影子=衰减率*参数之前的影子+（1-衰减率）*最新参数
	ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step) # MOVING_AVERAGE_DECAY 表示滑动平均衰减率，一般会赋接近 1 的值，global_step 表示当前训练了多少轮
	ema_op = ema.apply(tf.trainable_variables())  # 对所有待优化的参数求滑动平均。ema.apply()函数实现对括号内参数求滑动平均，tf.trainable_variables()函数实现把所有待训练参数汇总为列表。
	# ema_op = ema.apply(tf.Variable(0))
	with tf.control_dependencies([train_step,ema_op]): # 实现将滑动平均和训练过程同步运行
		train_op = tf.no_op(name='train')

	# 实例化saver对象。后面会调用saver.save()方法，向文件夹中写入包含当前模型中所有可训练变量的 checkpoint 文件。
	# saver.save()保存的文件有:(1).meta：保存了tensorflow计算图的结构，可以简单理解为神经网络的网络结构；(2).data文件：数据文件，保存变量的值(w、b等)；
	#                         (3).index文件：保存了当前参数名，通常不是必须的；(4)checkpoint文件：保存了目录下所有的模型文件列表。
	saver = tf.train.Saver()

	with tf.Session() as sess:
		# 初始化所有模型参数
		init_op = tf.global_variables_initializer()
		sess.run(init_op)

		#断点续训
		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH) # 如果断点文件夹中包含有效断点状态文件，则返回该文件
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess,ckpt.model_checkpoint_path) # 恢复当前会话，将ckpt中的值赋给w和b

		# 利用 sess.run( )函数实现模型的训练优化过程，并每间隔一定轮数保存一次模型。
		for i in range(STEPS):
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			_,loss_value,step = sess.run([train_op,loss,global_step], feed_dict={x:xs,y_:ys}) # 开始训练模型
			# print(y, labels=tf.argmax(y_,1))
			if i%1000 == 0:
				print("After %d training step(s),loss on training batch is %g."%(step,loss_value))
				saver.save(sess, os.path.join(MODEL_SAVE_PATH,MODEL_NAME), global_step=global_step) # 将神经网络模型中所有的参数等信息保存到指定的路径中，并在存放网络模型的文件夹名称中注明保存模型时的训练轮数

def main():
	# 使用input_data模块中的read_data_sets()函数加载mnist数据集
	mnist = input_data.read_data_sets("./data/", one_hot=True) # 第一个参数表示数据集存放路径，第二个参数表示数据集的存取形式。one_hot=True表示以独热码形式存取数据集
	backward(mnist)

if __name__ == '__main__':
	main()

