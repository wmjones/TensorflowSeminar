# Part 8 Tensorboard
import time
import tensorflow as tf
import numpy as np
batch_size = 10


def create_batch():
    x = np.random.rand(batch_size, 2)*10
    y = np.sin(x[:, 0]) + np.sin(x[:, 1])
    y = y.reshape(-1, 1)
    return(x, y)


MAX_STEPS = 100000
g = tf.Graph()
with g.as_default():
    sess = tf.Session(graph=g)
    inputs = tf.placeholder(tf.float32, [None, 2])
    labels = tf.placeholder(tf.float32, [None, 1])
    layer1 = tf.layers.dense(inputs, 10, activation=tf.nn.relu)
    layer2 = tf.layers.dense(layer1, 10, activation=tf.nn.relu)
    layer3 = tf.layers.dense(layer2, 10, activation=tf.nn.relu)
    layer4 = tf.layers.dense(layer3, 10, activation=tf.nn.relu)
    output = tf.layers.dense(layer4, 1)
    loss = tf.losses.mean_squared_error(output, labels)
    global_step = tf.Variable(0, trainable=False, name="step")
    train_opt = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss, global_step=global_step)

    # Tensorboard
    log_writer = tf.summary.FileWriter("logs/" + str(time.time()))
    log_writer.add_graph(sess.graph)
    tf.summary.scalar("MSE_Loss", loss)

    normal_dist = tf.distributions.Normal(0.0, 1.0)
    normal_dist_sample = tf.Variable(normal_dist.sample([100]))
    tf.summary.histogram("Practice_Histogram", normal_dist_sample)
    merged = tf.summary.merge_all()

    init = tf.global_variables_initializer()

sess.run(init)
for step in range(MAX_STEPS):
    x, y = create_batch()
    feed_dict = {inputs: x, labels: y}
    train_step = sess.run(global_step)
    sess.run(train_opt, feed_dict=feed_dict)
    if step % 1000 == 0:
        training_loss, _, _, summary = sess.run([loss, train_opt, normal_dist_sample, merged], feed_dict=feed_dict)
        log_writer.add_summary(summary, train_step)
        print(training_loss)

log_writer.close()
