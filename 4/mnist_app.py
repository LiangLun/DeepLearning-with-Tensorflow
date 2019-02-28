# encoding:utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_backward
import mnist_forward


# 对图像进行预处理，包括resize、转变灰度图、二值化操作
def pre_pic(picName):
	img = Image.open(picName)
	reIm = img.resize((28, 28), Image.ANTIALIAS) # 将图像大小改为28*28，Image.ANTIALIAS指高质量
	im_arr = np.array(reIm.convert('L')) #变成灰度图，并转化为矩阵
	threshold = 50 # 设定合理的阈值
	for i in range(28):
		for j in range(28):
			im_arr[i][j] = 255 - im_arr[i][j] # 将白底黑字变成输入要求的黑底白字
			if(im_arr[i][j] < threshold):  # 对图像做二值化处理
				im_arr[i][j] = 0
			else:
				im_arr[i][j] = 255
	nm_arr = im_arr.reshape([1, 784]) # 把图像拉成1行784列
	nm_arr = nm_arr.astype(np.float32) # 把值变为浮点型
	img_ready = np.multiply(nm_arr, 1.0 / 255.0)
	return img_ready

def restore_model(testPicArr):
	# 创建一个默认图，在该图下执行以下操作
	with tf.Graph().as_default() as tg:
		x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
		y = mnist_forward.forward(x, None)
		preValue = tf.argmax(y, 1) #得到最大的预测值

		# 实现滑动平均模型，参数MOVING_AVERAGE_DECAY用于控制模型更新的速度，训练过程中会对每个变量
		# 维护一个影子变量，这个影子变量就是相对应变量的初始值，每次变量更新时，影子变量就会随之更新
		variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		with tf.Session() as sess:
			#通过checkpoint文件定位到最新保存的模型
			ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess,ckpt.model_checkpoint_path)

				preValue = sess.run(preValue, feed_dict={x:testPicArr})
				return preValue
			else:
				print("No check file found")
				return -1


def application():
	testNum = input("input the number of test pictures:")
	for i in range(int(testNum)):
		# testPic = raw_input("the path of test picture:") # python2 用法
		testPic = input("the path of test picture:")  # python3
		testPicArr = pre_pic(testPic)
		preValue = restore_model(testPicArr)
		print("The prediction number is :",preValue)


def main():
	application()

if __name__ == '__main__':
	main()		

