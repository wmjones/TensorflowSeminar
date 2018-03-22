# Part 1 "hello world"
import tensorflow as tf
hello = tf.constant("hello world")
sess = tf.Session()
print(sess.run(hello))

# # Part 2 "batch_size"
import numpy as np
batch_size = 10


def create_batch():
    x = np.random.rand(batch_size, 2)*10
    y = np.sin(x[:, 0]) + np.sin(x[:, 1])
    y = y.reshape(-1, 1)
    return(x, y)


print(create_batch())

# Part 3 http://playground.tensorflow.org/

# Part 4 Introduction to Classes and super()


class SimpleClass():
    def __init__(self):
        print("hello from simple class")

    def foo(self):
        print("hello from foo")


x = SimpleClass()
x.foo()
print(type(x))


class ExtendedSimpleClass(SimpleClass):
    def __init__(self):
        super().__init__()
        print("hello from extended class")


y = ExtendedSimpleClass()
y.foo()

# Part 5 creating a computational graph


class Graph():
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []

    def set_as_default(self):
        global _default_graph
        _default_graph = self


class Operation():
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []

        for node in input_nodes:
            node.output_nodes.append(self)

        _default_graph.operations.append(self)

    def compute(self):
        pass


class add(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var + y_var


class multiply(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var * y_var


class Placeholder():
    def __init__(self):
        self.output_nodes = []
        _default_graph.placeholders.append(self)


class Variable():
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []
        _default_graph.placeholders.append(self)


def traverse_postorder(operation):
    # not important to understand. This is just so that the nodes are computed in the correct order
    nodes_postorder = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)
    recurse(operation)
    return(nodes_postorder)


class Session():

    def run(self, operation, feed_dict={}):
        nodes_postorder = traverse_postorder(operation)
        for node in nodes_postorder:
            if type(node) == Placeholder:
                node.output = feed_dict[node]
            elif type(node) == Variable:
                node.output = node.value
            else:
                # Then it must be an operation
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)  # * so it can handle any length of inputs allow to add 3 items
            if type(node.output) == list:
                node.output = np.array(node.output)
        return operation.output


g = Graph()
g.set_as_default()
print(_default_graph)
A = Variable(10)
b = Variable(1)
x = Placeholder()
y = multiply(A, x)
z = add(y, b)

my_sess = Session()
result = my_sess.run(operation=z, feed_dict={x: 10})
print(result)


# Part 6 Introduction to Tensorflow

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

# Part 7 Introduction to Tensorflow API
# https://www.tensorflow.org/api_docs/
MAX_STEPS = 1000

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
sess.run(init)
for step in range(MAX_STEPS):
    x, y = create_batch()
    feed_dict = {inputs: x, labels: y}
    sess.run(train_opt, feed_dict=feed_dict)
    if step % 100 == 0:
        training_loss, _ = sess.run([loss, train_opt], feed_dict=feed_dict)
        print(training_loss)


# Part 8 Tensorboard
import time

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
