# !/usr/bin/env python3
# coding=utf-8

"""
LeNet-5 Using TensorFlow
Author : Chai Zheng, Ph.D.@Zhejiang University, Hangzhou
Email  : zchaizju@gmail.com
Blog   : http://blog.csdn.net/chai_zheng/
Github : https://github.com/Chai-Zheng/
Date   : 2018.3.29
"""

import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder('float', shape=[None, 28*28])
y_true = tf.placeholder('float', shape=[None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])


def weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 1st layer: conv+relu+max_pool
w_conv1 = weights([5, 5, 1, 6])
b_conv1 = bias([6])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 2nd layer: conv+relu+max_pool
w_conv2 = weights([5, 5, 6, 16])
b_conv2 = bias([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])

# 3rd layer: 3*full connection
w_fc1 = weights([7*7*16, 120])
b_fc1 = bias([120])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1)+b_fc1)

w_fc2 = weights([120, 84])
b_fc2 = bias([84])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2)+b_fc2)

w_fc3 = weights([84, 10])
b_fc3 = bias([10])
h_fc3 = tf.nn.softmax(tf.matmul(h_fc2, w_fc3)+b_fc3)

cross_entropy = -tf.reduce_sum(y_true*tf.log(h_fc3))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(h_fc3, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch = mnist.train.next_batch(60)
    if i%100 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch[0], y_true: batch[1]})
        print('step {}, training accuracy: {}'.format(i, train_accuracy))
    train_step.run(session=sess, feed_dict={x: batch[0], y_true: batch[1]})

print('test accuracy: {}'.format(accuracy.eval(session=sess, feed_dict={x: mnist.test.images, y_true:
    mnist.test.labels})))
