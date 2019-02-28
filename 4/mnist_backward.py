# encoding:utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os
import mnist_generateds # 不同点1

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="./model/"
MODEL_NAME="mnist_model"
train_num_examples = 60000 # 不同点2 mnist.train.num_examples

# def backward(mnist):
def backward():
    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
    y = mnist_forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False) 

    # 定义损失函数
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    # 指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train_num_examples / BATCH_SIZE, # mnist.train.num_examples/BATCH_SIZE,
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
    # 获取batch_size张图片和标签
    img_batch, label_batch = mnist_generateds.get_tfrecord(BATCH_SIZE, isTrain=True) # 不同点3 用函数get_tfrecord，一次批获取batch_size张图片和标签 

    with tf.Session() as sess:
        # 初始化参数
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        #断点续训
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
		
        # 开启线程协调器 	
        coord = tf.train.Coordinator() # 不同点4
        threads = tf.train.start_queue_runners(sess=sess, coord=coord) # 不同点5 该句将会启动输入队列的线程，填充训练样本到队列中，以便出队操作可以从队列中拿到样本。
        
        for i in range(STEPS):
            xs, ys = sess.run([img_batch, label_batch])#不同点6 执行图片和标签的批获取
            # xs,ys=mnist.train.next_batch(BATCH_SIZE) #直接读出图片和标签
            # 开始训练
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
        # 关闭线程协调器
        coord.request_stop() # 不同点7
        coord.join(threads) # 不同点8


def main():
    # mnist=input_data.read_data_sets("./data/",one_hot=True)
    # backward(mnist)
    backward() # 不同点9

if __name__ == '__main__':
    main()


