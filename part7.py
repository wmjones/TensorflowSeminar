# Part 7 Introduction to Tensorflow API
# https://www.tensorflow.org/api_docs/
import tensorflow as tf
import numpy as np
batch_size = 10
MAX_STEPS = 1000


def create_batch():
    x = np.random.rand(batch_size, 2)*10
    y = np.sin(x[:, 0]) + np.sin(x[:, 1])
    y = y.reshape(-1, 1)
    return(x, y)


inputs = tf.placeholder(tf.float32, [None, 2])
labels = tf.placeholder(tf.float32, [None, 1])
layer1 = tf.layers.dense(inputs, 10)
layer2 = tf.layers.dense(layer1, 10)
layer3 = tf.layers.dense(layer2, 10)
layer4 = tf.layers.dense(layer3, 10)
output = tf.layers.dense(layer4, 1)
loss = tf.losses.mean_squared_error(output, labels)
global_step = tf.Variable(0, trainable=False, name="step")
train_opt = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for step in range(MAX_STEPS):
    x, y = create_batch()
    feed_dict = {inputs: x, labels: y}
    sess.run(train_opt, feed_dict=feed_dict)
    if step % 100 == 0:
        training_loss, _ = sess.run([loss, train_opt], feed_dict=feed_dict)
        print(training_loss)
