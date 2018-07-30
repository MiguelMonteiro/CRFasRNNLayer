import tensorflow as tf
import numpy as np
from PIL import Image
from os import sys, path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import lattice_filter_op_loader
from crf_rnn_layer import crf_rnn_layer

module = lattice_filter_op_loader.module

rgb = np.array(Image.open('image.jpg'))

gt = np.array(Image.open('segmentation.png'))
gt = np.sum(gt[:, :, :-1], axis=-1)
ground_truth = np.stack((gt == 0, gt != 0), axis=-1).astype(np.uint8)

ns = np.array(Image.open('noisy_segmentation.png'))
ns = np.sum(ns[:, :, :-1], axis=-1)
noisy_segmentation = np.stack((ns == 0, ns != 0), axis=-1).astype(np.uint8)

unaries = tf.constant(np.expand_dims(noisy_segmentation, axis=0), dtype=tf.float32) * 255
reference_image = tf.constant(np.expand_dims(rgb, axis=0), dtype=tf.float32) / 255.0
ground_truth = tf.constant(np.expand_dims(ground_truth, axis=0), dtype=tf.float32)

num_classes = 2
theta_alpha = 20
theta_beta = 0.2
theta_gamma = 20
num_iterations = 5

with tf.device('gpu:0'):

    logits = crf_rnn_layer(unaries, reference_image, num_classes, theta_alpha, theta_beta, theta_gamma, num_iterations)

    filter_test = module.lattice_filter(unaries, reference_image, bilateral=True, theta_alpha=theta_alpha, theta_beta=theta_beta)
    filter_test = tf.unstack(tf.squeeze(tf.nn.softmax(filter_test)), axis=-1)[-1] * 255

    output_image = tf.round(tf.unstack(tf.squeeze(tf.nn.softmax(logits)), axis=-1)[-1]) * 255
    #output_image = tf.unstack(tf.squeeze(tf.nn.softmax(logits)), axis=-1)[-1] * 255

    epochs = 1000
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=ground_truth))
    train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        f = sess.run(filter_test)
        for i in range(epochs):
            _, o, l = sess.run([train, output_image, loss])
            print(i, l)

im = Image.fromarray(o.astype(np.uint8))
im.save('output_image.bmp')

im = Image.fromarray(f.astype(np.uint8))
im.save('filter_test.bmp')


