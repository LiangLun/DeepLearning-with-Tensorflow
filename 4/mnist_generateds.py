# encoding:utf-8

'''将数据集（训练图片和测试图片）转换成tfrecords文件
   tfrecords文件:是一种二进制文件，可先将图片和标签制作成该格式的文件。使用tfrecords进行数据读取，会提高内存利用率。
'''
import tensorflow as tf
import numpy as np
from PIL import Image
import os


image_train_path = './mnist_data_jpg/mnist_train_jpg_60000/'
label_train_path = 'mnist_data_jpg/mnist_train_jpg_60000.txt'
tfRecord_train = './data/mnist_train.tfrecords'
image_test_path = './mnist_data_jpg/mnist_test_jpg_10000/'
label_test_path = './mnist_data_jpg/mnist_test_jpg_10000.txt'
tfRecord_test = './data/mnist_test.tfrecords'
data_path = './data'
resize_height = 28
resize_width = 28

# 用txt文件生成tfrecords文件
def write_tfRecord(tfRecordName, image_path, label_path):
	writer = tf.python_io.TFRecordWriter(tfRecordName) #新建一个writer。tfRecordName为要写入的tfrecords文件
	num_pic = 0  #计数器
	f = open(label_path, 'r')
	contents = f.readlines()  # readlines()方法用于读取所有行(直到结束符 EOF)并返回列表list。可用for..in..来读取 
	f.close()
	# 循环遍历每张图和标签
	for content in contents:
		value = content.split()  # 以空格分隔字符串，变成数组list（这里的括号中的空格可加可不加）
		img_path = image_path + value[0] # 获得图片路径
		img = Image.open(img_path)
		img_raw = img.tobytes()  # 将img转化成2进制数据
		labels = [0] * 10   # 将labels中10元素赋值为0，即labels=[0,0,0,0,0,0,0,0,0,0]
		labels[int(value[1])] = 1   # 制作标签，将labels对应位上的数赋值为1
		# 把每张图片和标签封装到example中。tf.train.Example:用来存储训练数据。训练数据的特征用键值对的形式表示。
		example = tf.train.Example(features=tf.train.Features(feature={
			'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),  # 字符串（BytesList），其中value=后要为一个list
			'label':tf.train.Feature(int64_list=tf.train.Int64List(value=labels))  # 整数列表（Int64List），其中value=后要为一个list
			}))
		writer.write(example.SerializeToString())  # 把example进行序列化。SerializeToString():把数据序列化成字符串存储
		num_pic += 1 #每保存一张图片计数器加一
		print("the number of picture:",num_pic) #打印进度提示
	writer.close()
	print("write tfrecord sucessful")

# 生成tfrecords文件
def generate_tfRecord():
	isExists = os.path.exists(data_path)
	if not isExists:
		os.makedirs(data_path)
		print("the directory was created sucessfully")
	else:
		print("directory already exists")
	write_tfRecord(tfRecord_train, image_train_path, label_train_path) 
	write_tfRecord(tfRecord_test, image_test_path, label_test_path)


def read_tfRecord(tfRecored_path):
	filename_queue = tf.train.string_input_producer([tfRecored_path]) #生成一个先入先出的队列
	reader = tf.TFRecordReader()  #新建一个reader
	# 解序列化。把读出的每个样本保存在serialized_example中进行解序列化，标签和图片的键名应该和制作 tfrecords 的键名相同，其中标签给出几分类。
	_, serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example,  # 该函数可以将 tf.train.Example 协议内存块(protocol buffer)解析为张量
		features={
		'label':tf.FixedLenFeature([10], tf.int64),  # 10表示10分类
		'img_raw':tf.FixedLenFeature([], tf.string)
		})
	img = tf.decode_raw(features['img_raw'], tf.uint8)  # 将img_raw转换8位无符号整型
	img.set_shape([784])  # 把形状变为一行784列
	img = tf.cast(img, tf.float32) * (1.0 / 255)  # 变成0-1的浮点数
	label = tf.cast(features['label'],tf.float32)  # 把标签列表变为浮点数
	return img, label

# 解析tfrecords文件,获取训练集或者测试集中的图片和标签
def get_tfrecord(num, isTrain=True):#num是一次读取的数量，istrain是看读取的是训练集还是测试集
	if isTrain:
		tfRecored_path = tfRecord_train
	else:
		tfRecored_path = tfRecord_test
	img, label = read_tfRecord(tfRecored_path)
	# 随机读取一个batch的数据
	img_batch,label_batch = tf.train.shuffle_batch([img, label],
		batch_size=num,
		num_threads=2,
		capacity=1000,
		min_after_dequeue=700)
	return img_batch, label_batch # 返回的图片和标签为随机抽取的batch_size组

def main():
	generate_tfRecord()

if __name__ == '__main__':
	main()