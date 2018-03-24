# Part 6 Introduction to Tensorflow
import tensorflow as tf
import numpy as np
batch_size = 10


def create_batch():
    x = np.random.rand(batch_size, 2)*10
    y = np.sin(x[:, 0]) + np.sin(x[:, 1])
    y = y.reshape(-1, 1)
    return(x, y)


A = tf.Variable(10, dtype=tf.float32)
b = tf.Variable(1, dtype=tf.float32)
x = tf.placeholder(dtype=tf.float32)
y = tf.multiply(A, x)
z = tf.add(y, b)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
feed_dict = {x: 10}
print(sess.run([z], feed_dict=feed_dict))
