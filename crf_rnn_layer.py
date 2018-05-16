"""
MIT License

Copyright (c) 2017 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import tensorflow as tf
import lattice_filter_op_loader

module = lattice_filter_op_loader.module


def crf_rnn_layer(unaries, reference_image, num_classes, theta_alpha, theta_beta, theta_gamma, num_iterations):
    with tf.variable_scope('crf_as_rnn_layer'):
        spatial_ker_weights = tf.get_variable('spatial_ker_weights', shape=(num_classes), initializer=tf.initializers.truncated_normal(mean=0, stddev=0.1))
        bilateral_ker_weights = tf.get_variable('bilateral_ker_weights', shape=(num_classes), initializer=tf.initializers.truncated_normal(mean=0, stddev=0.1))
        spatial_ker_weights = tf.diag(spatial_ker_weights)
        bilateral_ker_weights = tf.diag(bilateral_ker_weights)
        compatibility_matrix = tf.get_variable('compatibility_matrix', shape=(num_classes, num_classes), initializer=tf.initializers.truncated_normal(mean=0, stddev=0.1))

        # Prepare filter normalization coefficients
        unaries_shape = unaries.get_shape()
        all_ones = np.ones(unaries_shape, dtype=np.float32)
        spatial_norm_vals = module.lattice_filter(all_ones, reference_image, bilateral=False, theta_gamma=theta_gamma)
        bilateral_norm_vals = module.lattice_filter(all_ones, reference_image,
                                                    bilateral=True, theta_alpha=theta_alpha, theta_beta=theta_beta)
        q_values = unaries
        for i in range(num_iterations):

            q_values = tf.nn.softmax(q_values)

            # Spatial filtering
            spatial_out = module.lattice_filter(q_values, reference_image, bilateral=False, theta_gamma=theta_gamma)
            spatial_out = spatial_out / spatial_norm_vals

            # Bilateral filtering
            bilateral_out = module.lattice_filter(q_values, reference_image, bilateral=True, theta_alpha=theta_alpha,
                                                  theta_beta=theta_beta)
            bilateral_out = bilateral_out / bilateral_norm_vals

            # Weighting filter outputs
            message_passing = tf.matmul(spatial_ker_weights, tf.transpose(tf.reshape(spatial_out, (-1, num_classes)))) + \
                              tf.matmul(bilateral_ker_weights, tf.transpose(tf.reshape(bilateral_out, (-1, num_classes))))

            # Compatibility transform
            pairwise = tf.matmul(compatibility_matrix, message_passing)

            # Adding unary potentials
            pairwise = tf.reshape(tf.transpose(pairwise), unaries_shape)
            q_values = unaries - pairwise

        return q_values
