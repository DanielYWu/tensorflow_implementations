from __future__ import print_function


import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
from tensorflow.contrib import rnn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
mnist = input_data.read_data_sets('/data/mnist', one_hot=True)

learning_rate = 0.01
batch_size = 128
n_epochs = 25
display_step = 200

num_classes = 10

X = tf.placeholder("float", [None, 28, 28], name='X_placeholder')
Y = tf.placeholder("float", [None, 10], name='Y_placeholder')

# Define weights
with tf.name_scope("hidden"):
    weights = {
        'out': tf.Variable(tf.random_normal([128, 10]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([10]))
    }
    tf.summary.histogram("weights", weights['out'])
    tf.summary.histogram("biases", biases['out'])

X_timestamp = tf.unstack(X, 28, 1)

cell = rnn.BasicLSTMCell(128, forget_bias=1.0)

outputs, states = rnn.static_rnn(cell, X_timestamp, dtype=tf.float32)

logits = tf.matmul(outputs[-1], weights['out']) + biases['out']
tf.summary.histogram("activations", logits)

with tf.name_scope('cost'):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    tf.summary.scalar('cost', loss)

with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


with tf.name_scope('accuracy'):
    correct_pred = tf.equal(
        tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(
        tf.cast(correct_pred, tf.float32, name='accuracy'))
    tf.summary.scalar('accuracy', accuracy)


summary_op = tf.summary.merge_all()

with tf.Session() as sess:

    # Run the initializer
    start_time = time.time()
    writer = tf.summary.FileWriter('./graphs/lstm/', sess.graph)
    sess.run(tf.global_variables_initializer())

    for i in range(1000):

        X_batch, Y_batch = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        X_batch = X_batch.reshape((batch_size, 28, 28))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={X: X_batch, Y: Y_batch})

        if i % display_step == 0 or i == 1:
            # Calculate batch loss and accuracy
            loss_op, acc, summary = sess.run(
                [loss, accuracy, summary_op], feed_dict={X: X_batch, Y: Y_batch})
            writer.add_summary(summary, i)
            print("Step " + str(i) + ", Minibatch Loss= " +
                  "{:.4f}".format(loss_op) + ", Training Accuracy= " +
                  "{:.3f}".format(acc))
    print('Total time: {0} seconds'.format(time.time() - start_time))
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape(
        (-1, 28, 28))
    test_label = mnist.test.labels[:test_len]

    print("Testing Accuracy:",
          sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
