#coding:utf-8
'''
Lenet神经网络是Yann Lecun等人在1998年提出的。
大致结构为输入-卷积-激活-池化-卷积-激活-池化-全连接-输出
'''

import tensorflow as tf


IMAGE_SIZE = 28
NUM_CHANNELS = 1
CONV1_SIZE = 5
CONV1_KERNEL_NUM = 32
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64
FC_SIZE = 512
OUTPUT_NODE = 10

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w)) 
    return w

def get_bias(shape): 
    b = tf.Variable(tf.zeros(shape))  
    return b

# 卷积层，其中x是输入描述，w是卷积核描述，strides是卷积核滑动步长，padding表示是否填充0(VALID表示不填充，SAME表示填充)
# x的格式为[batch,图片行分辨率，列分辨率，通道数]，w为[行分辨率，列分辨率，通道数，核个数],strides为[1,行步长,列步长,1] 第1个和第4个元素固定为1
def conv2d(x,w):  
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

# 池化层，其中x是输入描述，ksize为池化核描述(仅有大小)，strides是池化核滑动步长，padding表示是否填充0
# x的格式为[batch, 图片行分辨率，列分辨率，通道数]，ksize为[1, 行分辨率，列分辨率，1] 第1个和第4个元素固定为1，strides为[1,行步长,列步长,1]
def max_pool_2x2(x): 
    # 最大池化用tf.nn.max_pool，均值池化用tf.nn.avg_pool
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 

# CNN的主要模块有卷积(convolutional)、激活(activation)、池化(pooling)、全连接(FC)
def forward(x, train, regularizer):
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer) # 卷积核5*5*1*32，16个卷积核，卷积操作输出的图片通道数是16
    conv1_b = get_bias([CONV1_KERNEL_NUM]) 
    conv1 = conv2d(x, conv1_w) 
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b)) # tf.nn.bias_add(value,bias,name=None)是tf.add的一个特例，其中bias必须是一维的
    pool1 = max_pool_2x2(relu1) # 2*2

    conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM],regularizer) # 5*5*32*64
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w) 
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2) # 2*2

    # 拉直：将pool2输出的矩阵形式转化成全连接层的输入格式即向量形式
    pool_shape = pool2.get_shape().as_list() # 得到pool2输出矩阵的维度，并存入list中
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3] # pool_shape[1]到[3]依次是矩阵的长、宽、深度，相乘得到矩阵拉长后的长度
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes]) # pool_shape[0]为一个batch值。将pool2转换为一个batch的向量再传入后续的全连接

    fc1_w = get_weight([nodes, FC_SIZE], regularizer) # nodes为FC输入层节点数
    fc1_b = get_bias([FC_SIZE]) 
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    # dropout 在神经网络训练过程中，将一部分神经元按照一定概率从神经网络中暂时舍弃。使用时被舍弃的神经元恢复链接。
    # dropout可以减少训练过程中过多参数，能够有效减少过拟合
    # 在实际应用中，用常常在前向传播构建神经网络时使用dropout来减小过拟合加快模型的训练速度。 一般放在全连接网站中
    if train:
        fc1 = tf.nn.dropout(fc1, 0.5) # tf.nn.dropout(上层输出，暂时舍弃神经元的概率)，会有指定概率的神经元被随机置零，置零的神经元不参与当前轮的参数优化。参数

    fc2_w = get_weight([FC_SIZE, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(fc1, fc2_w) + fc2_b
    return y 
