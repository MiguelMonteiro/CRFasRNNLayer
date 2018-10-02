import tensorflow as tf
import numpy as np
from crf_rnn_layer import crf_rnn_layer


def get_spatial_rank(x):
    """

    :param x: an input tensor with shape [batch_size, ..., num_channels]
    :return: the spatial rank of the tensor i.e. the number of spatial dimensions between batch_size and num_channels
    """
    return len(x.get_shape()) - 2


def get_num_channels(x):
    """

    :param x: an input tensor with shape [batch_size, ..., num_channels]
    :return: the number of channels of x
    """
    return int(x.get_shape()[-1])


def get_spatial_size(x):
    """

    :param x: an input tensor with shape [batch_size, ..., num_channels]
    :return: The spatial shape of x, excluding batch_size and num_channels.
    """
    return x.get_shape()[1:-1]


def constant_initializer(value, shape, lambda_initializer=True):
    if lambda_initializer:
        return np.full(shape, value).astype(np.float32)
    else:
        return tf.constant(value, tf.float32, shape)


def xavier_initializer_convolution(shape, dist='uniform', lambda_initializer=True):
    """
    Xavier initializer for N-D convolution patches. input_activations = patch_volume * in_channels;
    output_activations = patch_volume * out_channels; Uniform: lim = sqrt(3/(input_activations + output_activations))
    Normal: stddev =  sqrt(6/(input_activations + output_activations))
    :param shape: The shape of the convolution patch i.e. spatial_shape + [input_channels, output_channels]. The order of
    input_channels and output_channels is irrelevant, hence this can be used to initialize deconvolution parameters.
    :param dist: A string either 'uniform' or 'normal' determining the type of distribution
    :param lambda_initializer: Whether to return the initial actual values of the parameters (True) or placeholders that
    are initialized when the session is initiated
    :return: A numpy araray with the initial values for the parameters in the patch
    """
    s = len(shape) - 2
    num_activations = np.prod(shape[:s]) * np.sum(shape[s:])  # input_activations + output_activations
    if dist == 'uniform':
        lim = np.sqrt(6. / num_activations)
        if lambda_initializer:
            return np.random.uniform(-lim, lim, shape).astype(np.float32)
        else:
            return tf.random_uniform(shape, minval=-lim, maxval=lim)
    if dist == 'normal':
        stddev = np.sqrt(3. / num_activations)
        if lambda_initializer:
            return np.random.normal(0, stddev, shape).astype(np.float32)
        else:
            tf.truncated_normal(shape, mean=0, stddev=stddev)
    raise ValueError('Distribution must be either "uniform" or "normal".')


def convolution(x, filter, padding='SAME', strides=None, dilation_rate=None):
    w = tf.get_variable(name='weights', initializer=xavier_initializer_convolution(shape=filter))
    b = tf.get_variable(name='biases', initializer=constant_initializer(0, shape=filter[-1]))

    return tf.nn.convolution(x, w, padding, strides, dilation_rate) + b


def deconvolution(x, filter, output_shape, strides, padding='SAME'):
    w = tf.get_variable(name='weights', initializer=xavier_initializer_convolution(shape=filter))
    b = tf.get_variable(name='biases', initializer=constant_initializer(0, shape=filter[-2]))

    spatial_rank = get_spatial_rank(x)
    if spatial_rank == 2:
        return tf.nn.conv2d_transpose(x, w, output_shape, strides, padding) + b
    if spatial_rank == 3:
        return tf.nn.conv3d_transpose(x, w, output_shape, strides, padding) + b
    raise ValueError('Only 2D and 3D images supported.')


# down convolution
def down_convolution(x, factor, kernel_size):
    num_channels = get_num_channels(x)
    spatial_rank = get_spatial_rank(x)
    strides = spatial_rank * [factor]
    filter = spatial_rank * [kernel_size] + [num_channels, num_channels * factor]
    x = convolution(x, filter, strides=strides)
    return x


# up convolution
def up_convolution(x, output_shape, factor, kernel_size):
    num_channels = get_num_channels(x)
    spatial_rank = get_spatial_rank(x)
    strides = [1] + spatial_rank * [factor] + [1]
    filter = spatial_rank * [kernel_size] + [num_channels // factor, num_channels]
    x = deconvolution(x, filter, output_shape, strides=strides)
    return x


def convolution_block(layer_input, num_convolutions, keep_prob, activation_fn):
    n_channels = get_num_channels(layer_input)
    spatial_rank = get_spatial_rank(layer_input)
    x = layer_input
    kernel = spatial_rank * [5] + [n_channels, n_channels]
    for i in range(num_convolutions):
        with tf.variable_scope('conv_' + str(i + 1)):
            x = convolution(x, kernel)
            if i == num_convolutions - 1:
                x = x + layer_input
            x = activation_fn(x)
            x = tf.nn.dropout(x, keep_prob)
    return x


def convolution_block_2(layer_input, fine_grained_features, num_convolutions, keep_prob, activation_fn):
    n_channels = get_num_channels(layer_input)
    spatial_rank = get_spatial_rank(layer_input)
    x = tf.concat((layer_input, fine_grained_features), axis=-1)
    for i in range(0, num_convolutions):
        with tf.variable_scope('conv_' + str(i + 1)):
            kernel = spatial_rank * [5]
            kernel = kernel + [n_channels * 2, n_channels] if i == 0 else kernel + [n_channels, n_channels]
            x = convolution(x, kernel)
            if i == num_convolutions - 1:
                x = x + layer_input
            x = activation_fn(x)
            x = tf.nn.dropout(x, keep_prob)
    return x


class VNetCRF(object):
    def __init__(self,
                 num_classes,
                 keep_prob=1.0,
                 num_channels=16,
                 num_levels=4,
                 num_convolutions=(1, 2, 3, 3),
                 bottom_convolutions=3,
                 activation_fn=tf.nn.relu,
                 theta_alpha=50,
                 theta_beta=25,
                 theta_gamma=50,
                 num_iterations=5):
        """
        Implements VNet architecture https://arxiv.org/abs/1606.04797
        :param num_classes: Number of output classes.
        :param keep_prob: Dropout keep probability, set to 1.0 if not training or if no dropout is desired.
        :param num_channels: The number of output channels in the first level, this will be doubled every level.
        :param num_levels: The number of levels in the network. Default is 4 as in the paper.
        :param num_convolutions: An array with the number of convolutions at each level.
        :param bottom_convolutions: The number of convolutions at the bottom level of the network.
        :param activation_fn: The activation function.
        :param theta_alpha: Spatial standard deviation for bilateral filter
        :param theta_beta: Color standard deviation for bilateral filter
        :param theta_gamma: Spatial standard deviation for Gaussian filter
        :param num_iterations: Number of iterations for mean field approximation of the CRF
        """
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        self.num_channels = num_channels
        assert num_levels == len(num_convolutions)
        self.num_levels = num_levels
        self.num_convolutions = num_convolutions
        self.bottom_convolutions = bottom_convolutions
        self.activation_fn = activation_fn
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations

    def network_fn(self, x, is_training):

        input_image = x
        input_channels = get_num_channels(x)
        spatial_rank = get_spatial_rank(x)
        keep_prob = self.keep_prob if is_training else 1.0
        # if the input has more than 1 channel it has to be expanded because broadcasting only works for 1 input
        # channel
        with tf.variable_scope('vnet/input_layer'):
            if input_channels == 1:
                x = tf.tile(x, (spatial_rank + 1) * [1] + [self.num_channels])
            else:
                x = self.activation_fn(convolution(x, spatial_rank * [5] + [input_channels, self.num_channels]))

        features = list()
        for l in range(self.num_levels):
            with tf.variable_scope('vnet/encoder/level_' + str(l + 1)):
                x = convolution_block(x, self.num_convolutions[l], keep_prob, activation_fn=self.activation_fn)
                features.append(x)
                with tf.variable_scope('down_convolution'):
                    x = self.activation_fn(down_convolution(x, factor=2, kernel_size=2))

        with tf.variable_scope('vnet/bottom_level'):
            x = convolution_block(x, self.bottom_convolutions, keep_prob, activation_fn=self.activation_fn)

        for l in reversed(range(self.num_levels)):
            with tf.variable_scope('vnet/decoder/level_' + str(l + 1)):
                f = features[l]
                with tf.variable_scope('up_convolution'):
                    x = self.activation_fn(up_convolution(x, tf.shape(f), factor=2, kernel_size=2))

                x = convolution_block_2(x, f, self.num_convolutions[l], keep_prob, activation_fn=self.activation_fn)

        with tf.variable_scope('vnet/output_layer'):
            logits = convolution(x, spatial_rank * [1] + [self.num_channels, self.num_classes])

        with tf.variable_scope('crf_as_rnn'):
            logits = crf_rnn_layer(unaries=logits,
                                   reference_image=input_image,
                                   num_classes=self.num_classes,
                                   theta_alpha=self.theta_alpha,
                                   theta_beta=self.theta_beta,
                                   theta_gamma=self.theta_gamma,
                                   num_iterations=self.num_iterations)

        return logits


def input_function(batch_size, reference_channels, num_classes):
    # dummy inputs (feed your own images by using TFRecordDataset: tf.data.TFRecordDataset(filenames))
    input_image = tf.constant(1.0, shape=(batch_size, 100, 100, 50, reference_channels), dtype=tf.float32)
    ground_truth = tf.constant(1.0, shape=(batch_size, 100, 100, 50, num_classes), dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensors((input_image, ground_truth))
    dataset = dataset.repeat(10)  # 10 epochs
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


if __name__ == "__main__":
    # Compile with SPATIAL_DIMENSIONS=3, REFERENCE_CHANNELS=4, INPUT_CHANNELS=2 (num_classes)
    BATCH_SIZE = 1
    REFERENCE_CHANNELS = 4
    INPUT_CHANNELS = 2
    num_classes = INPUT_CHANNELS

    with tf.Graph().as_default():

        input_image, ground_truth = input_function(BATCH_SIZE, REFERENCE_CHANNELS, num_classes)
        net = VNetCRF(num_classes=num_classes)
        logits = net.network_fn(input_image, is_training=True)
        logits = tf.reshape(logits, (-1, num_classes))
        labels = tf.reshape(ground_truth, (-1, num_classes))
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        probability = tf.nn.softmax(logits)
        prediction = tf.round(probability)
        # calculate dice coefficient and/or other metrics that are useful to you

        with tf.Session() as sess:
            while not sess.should_stop():
                _, l, p, = sess.run([train_op, loss, prediction])
                print('loss: %:.3f\n'.format(l))
