# encoding:utf-8

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
import mnist_generateds # 不同点1 

TEST_INTERVAL_SECS = 5
TEST_NUM = 10000 # 不同点2 mnist.test.num_examples

# def test(mnist):
def test():
	with tf.Graph().as_default() as g:
		x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
		y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
		y = mnist_forward.forward(x, None)

		# 加载模型中参数的滑动平均值
		ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
		ema_restore = ema.variables_to_restore()
		saver = tf.train.Saver(ema_restore)

		# 神经网络模型准确率评估
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		# 读取测试集图片
		img_batch, label_batch = mnist_generateds.get_tfrecord(TEST_NUM, isTrain=False) #不同点3

		while True:
			with tf.Session() as sess:
				# 神经网络模型的加载
				ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(sess, ckpt.model_checkpoint_path)
					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
					
					# 开启线程协调器
					coord = tf.train.Coordinator()
					threads = tf.train.start_queue_runners(sess=sess, coord=coord)

					xs,ys = sess.run([img_batch, label_batch])

					# accuracy_score = sess.run(accuracy, feed_dict={x:mnist.test.images,y_:mnist.test.labels})
					accuracy_score = sess.run(accuracy, feed_dict={x:xs, y_:ys})
					print("After %s training step(s),test accuracy = %g" % (global_step, accuracy_score))
					
					# 关闭线程协调器
					coord.request_stop()
					coord.join(threads)

				else:
					print("No checkpoint file found")
					return
			time.sleep(TEST_INTERVAL_SECS)				

def main():
	# mnist = input_data.read_data_sets("./data/", one_hot=True)
	# test(mnist)
	test()

if __name__ == '__main__':
	main()
