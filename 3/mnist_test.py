# encoding:utf-8

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward

TEST_INTERVAL_SECS = 5

def test(mnist):
	# tf.Graph( ).as_default( ) ) 函数表示将当前图设置成为默认图，并返回一个上下文管理器
	with tf.Graph().as_default() as g: # 将在 Graph()内定义的节点加入到计算图g中
		x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
		y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
		y = mnist_forward.forward(x, None)

		# 加载模型中参数的滑动平均值
		ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
		ema_restore = ema.variables_to_restore()
		saver = tf.train.Saver(ema_restore)
		# 神经网络模型准确率评估。
		#  tf.equal()函数判断预测结果张量和实际标签张量的每个维度是否相等，若相等则返回 True，不相等则返回 False
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) # tf.argmax()函数取出每张图片对应向量中最大值元素对应的索引值
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # tf.cast()函数将得到的布尔型数值转化为实数型

		while True:
			with tf.Session() as sess:
				# 神经网络模型的加载
				ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(sess, ckpt.model_checkpoint_path) # 将保存的神经网络模型加载到当前会话中

					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1] # 字符串.split()表示指定拆分符对字符串拆分
					accuracy_score = sess.run(accuracy, feed_dict = {x:mnist.test.images, y_:mnist.test.labels})
					print("After %s training step(s),test accuracy = %g" % (global_step, accuracy_score))
				else:
					print("No checkpoint file found")
					return
			time.sleep(TEST_INTERVAL_SECS)				

def main():
	mnist = input_data.read_data_sets("./data/", one_hot=True)
	test(mnist)

if __name__ == '__main__':
	main()
