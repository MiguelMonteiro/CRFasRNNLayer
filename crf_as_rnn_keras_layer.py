"""
MIT License

Copyright (c) 2017 Sadeep Jayasumana , Miguel Monteiro, Walter de Back

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

from keras.engine.topology import Layer
import tensorflow as tf
import lattice_filter_op_loader

module = lattice_filter_op_loader.module


class CRF_RNN_Layer(Layer):
    """ Implements the CRF-RNN layer.
    See https://github.com/sadeepj/crfasrnn_keras/blob/master/src/crfrnn_layer.py
    Based on GPU implementation here: https://github.com/MiguelMonteiro/CRFasRNNLayer

    Unaries and reference image must be provided in order: [unaries, ref_image]
    """

    def __init__(self,
                 image_dims,
                 num_classes,
                 theta_alpha,
                 theta_beta,
                 theta_gamma,
                 num_iterations,
                 **kwargs):
        self.image_dims = image_dims
        self.num_classes = num_classes
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations
        self.spatial_ker_weights = None
        self.bilateral_ker_weights = None
        self.compatibility_matrix = None
        super(CRF_RNN_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.spatial_ker_weights = self.add_weight(name='spatial_ker_weights',
                                                   shape=(self.num_classes,),
                                                   initializer=tf.initializers.truncated_normal(mean=0, stddev=0.1),
                                                   trainable=True)

        self.spatial_ker_weights = tf.diag(self.spatial_ker_weights)

        self.bilateral_ker_weights = self.add_weight(name='bilateral_ker_weights',
                                                     shape=(self.num_classes,),
                                                     initializer=tf.initializers.truncated_normal(mean=0, stddev=0.1),
                                                     trainable=True)
        self.bilateral_ker_weights = tf.diag(self.bilateral_ker_weights)

        self.compatibility_matrix = self.add_weight(name='compatibility_matrix',
                                                    shape=(self.num_classes, self.num_classes),
                                                    initializer=tf.initializers.truncated_normal(mean=0, stddev=0.1),
                                                    trainable=True)

        super(CRF_RNN_Layer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        unaries = inputs[0]
        reference_image = inputs[1]

        # Prepare filter normalization coefficients
        unaries_shape = unaries.get_shape()
        q_values = unaries
        for i in range(self.num_iterations):
            q_values = tf.nn.softmax(q_values)

            # Spatial filtering
            spatial_out = module.lattice_filter(q_values, reference_image, bilateral=False,
                                                theta_gamma=self.theta_gamma)

            # Bilateral filtering
            bilateral_out = module.lattice_filter(q_values, reference_image, bilateral=True,
                                                  theta_alpha=self.theta_alpha, theta_beta=self.theta_beta)

            # Weighting filter outputs
            message_passing = tf.matmul(self.spatial_ker_weights,
                                        tf.transpose(tf.reshape(spatial_out, (-1, self.num_classes)))) + \
                              tf.matmul(self.bilateral_ker_weights,
                                        tf.transpose(tf.reshape(bilateral_out, (-1, self.num_classes))))

            # Compatibility transform
            pairwise = tf.matmul(self.compatibility_matrix, message_passing)

            # Adding unary potentials
            pairwise = tf.reshape(tf.transpose(pairwise), unaries_shape)
            q_values = unaries - pairwise

        return q_values

    def compute_output_shape(self, input_shape):
        return input_shape
