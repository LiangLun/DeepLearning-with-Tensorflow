# encoding:utf-8

'''示例1 ：输入为一维向量，输出0或者1。用于tesorflow在神经网络中的基本语法
   搭建神经网络的八股：第一步是准备数据；第二步是前向传播（定义输入、神经网络参数和输出）；
                     第三步是反向传播（定义损失函数和反向传播方法）；第四步是生成回话session，开始迭代训练
'''
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
seed = 23455

#基于seed产生随机数，若seed相同，产生数也相同
rng = np.random.RandomState(seed)
#随机数返回32行2列的矩阵，表示32组数据，作为输入训练集
X = rng.rand(32,2)
Y = [[int(x0+x1<1)] for (x0,x1) in X]
print ("X:\n",X)
print ("Y:\n",Y)
# for (x0,x1) in X:
# 	Y=[int(x0+x1<1)]
# print ("X:\n",X)
# print ("Y:\n",Y)

#-------1.定义神经网络的输入，参数和输出，定义前向传播过程-------
# placeholder机制用于提供输入数据。shape的第一维是None表示先空着，这样在feed_dict就可以喂入若干组数据了
x = tf.placeholder(tf.float32, shape=(None,2))
y_ = tf.placeholder(tf.float32, shape=(None,1))

# tf.Variable用于保存和更新神经网络中的参数，变量用这个。w1会产生2*3的数据，其中元素的均值为0，标准差为1 
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

# tf.matmul是矩阵乘法，不能直接用*。若用*，表示对应元素相乘
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#-------2.定义损失函数及反向传播方法-------
# 这里的loss用的是均方误差MMSE。tf.square是矩阵对应元素之差的平方，tf.reduce_mean是对矩阵中所有数求平均值
loss = tf.reduce_mean(tf.square(y-y_))
# 定义神经网络中反向传播的优化方法。tensorflow中常用的优化器支持7种优化器，常用的有三种：GradientDescentOptimizer、MomentOptimizer、AdamOptimizer。
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# train_step = tf.train.MomentOptimizer(0.001,0.9).minimize(loss)
# train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#-------3.生成对话，训练steps轮-------
# tensorflow中用回话(session)来执行定义好的运算
with tf.Session() as sess:
	# 变量初始化。用tf.global_variables_initializer()汇总所有变量，sess.run(init_op)则对所有变量初始化
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	# 输出未经训练的权重。在tensorflow中，若要打印一个张量a，用print(sess.run(a))；print(a)打印的是张量a的三个属性
	print("w1\n", sess.run(w1))
	print("w2\n", sess.run(w2))
	print("\n")

	#训练模型
	STEPS = 3000
	for i in range(STEPS):
		# 每次取8个数进行训练
		start = (i*BATCH_SIZE)%32
		end = start + BATCH_SIZE
		# 用feed_dict来指定训练数据x、y_的取值。feed_dict是一个字典(map)，字典中需要给出每个用到的placeholder的取值
		sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
		if i%500 ==0:
			# 计算指定步数的loss,因为loss是张量，要打印它，必须要sess.run，要sess.run计算loss必须要feed_dict
			total_loss = sess.run(loss, feed_dict={x:X,y_:Y})
			print("After %d training step(s),loss on all data is %g"%(i,total_loss))
			# print("After %d training step(s),loss on all data is %g"%(i, sess.run(loss)))
	# 输出训练后的参数取值
	print("\n")
	print("w1:\n", sess.run(w1))
	print("w2:\n", sess.run(w2))
	# print("w1:\n", w1)
	# print("w2:\n", w2)
