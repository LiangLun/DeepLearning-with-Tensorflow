#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_lenet5_forward
import os
import numpy as np

BATCH_SIZE = 100
LEARNING_RATE_BASE =  0.005 
LEARNING_RATE_DECAY = 0.99 
REGULARIZER = 0.0001 
STEPS = 50000 
MOVING_AVERAGE_DECAY = 0.99 
MODEL_SAVE_PATH = "./model/" 
MODEL_NAME = "mnist_model" 

def backward(mnist):
    x = tf.placeholder(tf.float32,[
    BATCH_SIZE,
    mnist_lenet5_forward.IMAGE_SIZE,
    mnist_lenet5_forward.IMAGE_SIZE,
    mnist_lenet5_forward.NUM_CHANNELS]) # 卷积层的输入是四阶张量
    y_ = tf.placeholder(tf.float32, [None, mnist_lenet5_forward.OUTPUT_NODE]) # 输出
    y = mnist_lenet5_forward.forward(x, True, REGULARIZER) 
    global_step = tf.Variable(0, trainable=False) 

    # 定义损失函数
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce) 
    loss = cem + tf.add_n(tf.get_collection('losses')) 

    # 指数衰减学习率
    learning_rate = tf.train.exponential_decay( 
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, 
        LEARNING_RATE_DECAY,
        staircase=True) 
    
    # 定义反向传播算法
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]): 
        train_op = tf.no_op(name='train')

    # 用于保存模型
    saver = tf.train.Saver() 

    with tf.Session() as sess: 
        # 初始化参数
        init_op = tf.global_variables_initializer() 
        sess.run(init_op) 

        # 断点续训
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH) 
        if ckpt and ckpt.model_checkpoint_path:
        	saver.restore(sess, ckpt.model_checkpoint_path) 

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE) 
            reshaped_xs = np.reshape(xs, (   # 读取一个batch数据，将输入数据xs转成与网络输入相同形状的矩阵。原来的xs是batch*784
            BATCH_SIZE,
            mnist_lenet5_forward.IMAGE_SIZE,
            mnist_lenet5_forward.IMAGE_SIZE,
            mnist_lenet5_forward.NUM_CHANNELS))
            # 喂入训练图像和标签，开始训练
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys}) 
            if i % 100 == 0: 
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True) 
    backward(mnist)

if __name__ == '__main__':
    main()


