import tensorflow as tf

class DeepQNetwork(object):
    def __init__(self, image_width, image_height, num_channels, num_actions, name):
        # network variables
        # conv1
        self.conv1_filter_size = 8
        self.conv1_filter_num = 32
        self.conv1_stride = 4
        self.conv1_output_size = (image_width - self.conv1_filter_size) / self.conv1_stride + 1    # 20 with 84 * 84 image and no padding
        self.conv1_weights, self.conv1_biases = \
            self.create_conv_net([self.conv1_filter_size, self.conv1_filter_size, num_channels, self.conv1_filter_num], name=name+'conv1')

        # conv2
        self.conv2_filter_size = 4
        self.conv2_filter_num = 64
        self.conv2_stride = 2
        self.conv2_output_size = (self.conv1_output_size - self.conv2_filter_size) / self.conv2_stride + 1    # 9 with 84 * 84 image no padding
        self.conv2_weights, self.conv2_biases = \
            self.create_conv_net([self.conv2_filter_size, self.conv2_filter_size, self.conv1_filter_num, self.conv2_filter_num], name=name+'conv2')

        # conv3
        self.conv3_filter_size = 3
        self.conv3_filter_num = 64
        self.conv3_stride = 1
        self.conv3_output_size = (self.conv2_output_size - self.conv3_filter_size) / self.conv3_stride + 1    # 7 with 84 * 84 image no padding
        self.conv3_weights, self.conv3_biases = \
            self.create_conv_net([self.conv3_filter_size, self.conv3_filter_size, self.conv2_filter_num, self.conv3_filter_num], name=name+'conv3')

        # inner product 1
        self.inner1_inputs = int(self.conv3_output_size * self.conv3_output_size * self.conv3_filter_num)    # should be 3136 for default
        self.inner1_outputs = 512
        self.inner1_weights, self.inner1_biases = self.create_inner_net([self.inner1_inputs, self.inner1_outputs], name=name+'inner1')

        # inner product 2
        self.inner2_inputs = self.inner1_outputs
        self.inner2_outputs = num_actions
        self.inner2_weights, self.inner2_biases = self.create_inner_net([self.inner2_inputs, self.inner2_outputs], name=name+'inner2')

        # Network variable saver
        self.saver = tf.train.Saver({var.name: var for var in self.weights_and_biases()})

        # Network name
        self.name = name


    def forward(self, state):
        conv1 = tf.nn.conv2d(state, self.conv1_weights, [1, self.conv1_stride, self.conv1_stride, 1], padding='VALID', name=self.name+'conv1')
        conv1 = tf.nn.relu(conv1 + self.conv1_biases, name=self.name+'relu1')
        conv2 = tf.nn.conv2d(conv1, self.conv2_weights, [1, self.conv2_stride, self.conv2_stride, 1], padding='VALID', name=self.name+'conv2')
        conv2 = tf.nn.relu(conv2 + self.conv2_biases, name=self.name+'relu2')
        conv3 = tf.nn.conv2d(conv2, self.conv3_weights, [1, self.conv3_stride, self.conv3_stride, 1], padding='VALID', name=self.name+'conv3')
        conv3 = tf.nn.relu(conv3 + self.conv3_biases, name=self.name+'relu3')
        shape = conv3.get_shape().as_list()
        reshape = tf.reshape(conv3, [shape[0], shape[1] * shape[2] * shape[3]])
        inner1 = tf.nn.relu(tf.matmul(reshape, self.inner1_weights) + self.inner1_biases)
        inner2 = tf.matmul(inner1, self.inner2_weights) + self.inner2_biases
        return inner2

    def q_values(self, state):
        return self.forward(state)

    def selected_q_values(self, state, action):
        return tf.mul(self.q_values(state), action)

    def clipped_loss(self, state, reward, action):
        filtered_qs = self.selected_q_values(state, action)
        error = tf.abs(reward - filtered_qs)
        quadratic = tf.clip_by_value(error, 0.0, 1.0)
        linear = error - quadratic
        return tf.reduce_sum(0.5 * tf.square(quadratic) + linear)

    def create_conv_net(self, shape, name):
        weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.01), name=name + 'weights')
        biases = tf.Variable(tf.constant(0.01, shape=[shape[3]]), name=name + 'biases')
        return weights, biases

    def create_inner_net(self, shape, name):
        weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.01), name=name + 'weights')
        biases = tf.Variable(tf.constant(0.01, shape=[shape[1]]), name=name + 'biases')
        return weights, biases

    def weights_and_biases(self):
        return [self.conv1_weights, self.conv1_biases,
                        self.conv2_weights, self.conv2_biases,
                        self.conv3_weights, self.conv3_biases,
                        self.inner1_weights, self.inner1_biases,
                        self.inner2_weights, self.inner2_biases]

    def copy_network_to(self, target, session):
        copy_operations = [target.assign(origin) for origin, target in zip(self.weights_and_biases(), target.weights_and_biases())]
        session.run(copy_operations)

    def save_parameters(self, session, file_name, global_step):
        self.saver.save(session, save_path=file_name, global_step=global_step)

    def restore_parameters(self, session, file_name):
        if file_name is None or file_name == '':
            print('Filename was not specified. Use default parameter')
            return
        self.saver.restore(session, save_path=file_name)

    def debug_print_variables(self, session):
        # show snippet of the weight for checking
        print(session.run(self.conv1_weights)[0])
