import tensorflow as tf
import numpy as np


class DCNN:
    def __init__(self, input_placeholder, label_placeholder):
        self.input = tf.expand_dims(input_placeholder, -1)
        true_output = label_placeholder
        # 两个卷积层
        convolution_output = self.convlayers(self.input)
        # 三个全连接层
        self.logits = self.fc_layers(convolution_output)
        # 计算交叉熵
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_output, logits=self.logits))
        # 配置Adam优化器
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        # 变量赋值
        tf.summary.scalar('cross_entropy', self.loss)
        # 定义变量作用域
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):

                correct_prediction = tf.equal(tf.argmax(true_output, 1), tf.argmax(self.logits, 1))
                self.correct_prediction = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))
            with tf.name_scope('accuracy'):
                # tf.cast转换数据类型
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
        tf.summary.scalar('accuracy', accuracy)

    @staticmethod
    def convlayers(input):
        #两个卷积层

        # zero-mean input
        with tf.name_scope('preprocess'):
           normalized_input = input
           #normalized_input = input - tf.expand_dims(tf.reduce_mean(input, 1), 2)
           #normalized_input = normalized_input / tf.expand_dims(tf.abs(tf.reduce_max(input, 2), 3))

        # conv1
        with tf.name_scope('conv1') as scope:
            # 初始化【32,1，8】的卷积核
            kernel = tf.Variable(tf.truncated_normal([32, 1, 8], dtype=tf.float64, stddev=1e-1), name='weights')
            conv = tf.nn.conv1d(normalized_input, kernel, 1, padding='SAME')
            #需要知道【３２，１，８】代表什么
            #初始化偏置项
            biases = tf.Variable(tf.constant(0.0, shape=[8], dtype=tf.float64), trainable=True, name='biases')
            #把偏置项添加到被卷积后的矩阵上
            preactivate = tf.nn.bias_add(conv, biases)
            #relu激活函数
            activation = tf.nn.relu(preactivate, name=scope)

        # conv2
        with tf.name_scope('conv2') as scope:
            # 需要知道【８，８，４】代表什么
            kernel = tf.Variable(tf.truncated_normal([8, 8, 4], dtype=tf.float64, stddev=1e-1), name='weights')
            conv = tf.nn.conv1d(activation, kernel, 1, padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[4], dtype=tf.float64), trainable=True, name='biases')
            preactivate = tf.nn.bias_add(conv, biases)
            activation = tf.nn.relu(preactivate, name=scope)
        return activation

    @staticmethod
    def fc_layers(conv):
        # fc1
        with tf.name_scope('fc1'):
            #get_shape()[1:])表示返回得到第二个之后的维度
            #np.prod返回行列的乘积
            shape = int(np.prod(conv.get_shape()[1:]))
            #初始化权重
            fc1w = tf.Variable(tf.truncated_normal([shape, 256], dtype=tf.float64, stddev=1e-1), name='weights')
            #初始化偏置
            fc1b = tf.Variable(tf.constant(1.0, shape=[256], dtype=tf.float64), trainable=True, name='biases')
            #把数据拉成每行是shape列的矩阵
            conv2_flat = tf.reshape(conv, [-1, shape])
            #和权重相乘再加上偏置
            fc1l = tf.nn.bias_add(tf.matmul(conv2_flat, fc1w), fc1b)
            fc1 = tf.nn.relu(fc1l)

        # fc2
        with tf.name_scope('fc2'):

            fc2w = tf.Variable(tf.truncated_normal([256, 64], dtype=tf.float64, stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[64], dtype=tf.float64), trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
            fc2 = tf.nn.relu(fc2l)

        # fc3
        with tf.name_scope('fc3'):
            fc3w = tf.Variable(tf.truncated_normal([64, 8], dtype=tf.float64, stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[8], dtype=tf.float64), trainable=True, name='biases')
            fc3 = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)

        # fc4
        with tf.name_scope('fc3'):
            fc4w = tf.Variable(tf.truncated_normal([8, 225], dtype=tf.float64, stddev=1e-1), name='weights')
            fc4b = tf.Variable(tf.constant(1.0, shape=[225], dtype=tf.float64), trainable=True, name='biases')
            logitstemp = tf.nn.bias_add(tf.matmul(fc3, fc4w), fc4b)
            logits = tf.nn.sigmoid(logitstemp)
        return logits

