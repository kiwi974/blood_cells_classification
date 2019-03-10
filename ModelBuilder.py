"""
    Class which allows to build a neural network, with its different training and 
    testing parameters. 
    The built network must be fed with grayscaled images. 
"""

import tensorflow as tf

class ModelBuilder:


    def __init__(self, input_height, input_width, classes, input_data, dropout1_rate, should_drop=True):
        """
        Build a ModelBuilder. 
        Parameters : 
            - input_height : height of the input images 
            - input_width : width of the input images 
            - classes : number of classes of the model 
            - input_data : the data fedding the model (a placeholder in practise)
            - dropout1_rate : dropout rate of the first dropout layer
            - mode : default is True for training, otherwise it is False for inference
        Return : 
            - logits : the built neural network 
        """
        self.input_height = input_height
        self.input_width = input_width
        self.classes = classes
        self.input_data = input_data
        self.dropout1_rate = dropout1_rate
        self.should_drop = should_drop



    

    def networkBuilder(self):
        """     
        Build the architecture of the convolutional network.            
        """
        with tf.variable_scope('input_layer', reuse=tf.AUTO_REUSE):
            # Build the input layer, with a dynamically computed batch size
            input_layer = tf.reshape(self.input_data, [-1, self.input_height, self.input_width, 1])

                        ###########################################
                        ########## CONVOLUTIONAL LAYER 1 ##########
                        ###########################################

        with tf.variable_scope('conv_1', reuse=tf.AUTO_REUSE):
            # Convolutional Layer 1
            conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=16,
                kernel_size=[3,3],
                padding="same",
                activation=tf.nn.relu
            )

        with tf.variable_scope('pool_1', reuse=tf.AUTO_REUSE):
            # Pooling Layer 1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        with tf.variable_scope('norm_1', reuse=tf.AUTO_REUSE):
            # Normalization Layer 1
            norm1 = tf.nn.local_response_normalization(input=tf.to_float(pool1))

                        ###########################################
                        ########## CONVOLUTIONAL LAYER 2 ##########
                        ###########################################

        with tf.variable_scope('conv_2', reuse=tf.AUTO_REUSE):
            # Convolutional Layer 2
            conv2 = tf.layers.conv2d(
                inputs=norm1,
                filters=16,
                kernel_size=[3,3],
                padding="same",
                activation=tf.nn.relu
            )

        with tf.variable_scope('pool_2', reuse=tf.AUTO_REUSE):
            # Pooling Layer 2
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        with tf.variable_scope('norm_2', reuse=tf.AUTO_REUSE):
            # Normalization Layer 2
            norm2 = tf.nn.local_response_normalization(input=tf.to_float(pool2))

                        ###########################################
                        ########## CONVOLUTIONAL LAYER 3 ##########
                        ###########################################

        with tf.variable_scope('conv_3', reuse=tf.AUTO_REUSE):
            # Convolutional Layer 3 
            conv3 = tf.layers.conv2d(
                inputs=norm2,
                filters=8,
                kernel_size=[3,3],
                padding="same",
                activation=tf.nn.relu
            )

        with tf.variable_scope('pool_3', reuse=tf.AUTO_REUSE):
            # Pooling Layer 3
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

        with tf.variable_scope('norm_3', reuse=tf.AUTO_REUSE):
            # Normalization Layer 3
            norm3 = tf.nn.local_response_normalization(input=tf.to_float(pool3))

                        ###########################################
                        ########## CONVOLUTIONAL LAYER 4 ##########
                        ###########################################

        with tf.variable_scope('conv_4', reuse=tf.AUTO_REUSE):
            # Convolutional Layer 3 
            conv4 = tf.layers.conv2d(
                inputs=norm3,
                filters=8,
                kernel_size=[3,3],
                padding="same",
                activation=tf.nn.relu
            )

        with tf.variable_scope('pool_4', reuse=tf.AUTO_REUSE):
            # Pooling Layer 3
            pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

        with tf.variable_scope('norm_4', reuse=tf.AUTO_REUSE):
            # Normalization Layer 3
            norm4 = tf.nn.local_response_normalization(input=tf.to_float(pool4))

                        #############################
                        ########## FLATTEN ##########
                        #############################

        with tf.variable_scope('pool_flat', reuse=tf.AUTO_REUSE):
            # Entry Flatten Layer 
            pool_flat = tf.reshape(norm4, [-1, 3 * 5 * 8]) 

                        ##################################
                        ########## DENSE LAYERS ##########
                        ##################################

        with tf.variable_scope('dense_1', reuse=tf.AUTO_REUSE):
            # Dense Layer 1
            dense1 = tf.layers.dense(inputs=pool_flat, units=16, activation=tf.nn.relu)

        with tf.variable_scope('dense_2', reuse=tf.AUTO_REUSE):
            # Dense Layer 2
            dense2 = tf.layers.dense(inputs=dense1, units=16, activation=tf.nn.relu)

        with tf.variable_scope('dense_3', reuse=tf.AUTO_REUSE):
            # Dense Layer 3
            dense3 = tf.layers.dense(inputs=dense2, units=8, activation=tf.nn.relu)

                        ###################################
                        ########## OUTPUT LOGITS ##########
                        ###################################
            
        with tf.variable_scope('logits', reuse=tf.AUTO_REUSE):
            # Logits Layer
            logits = tf.layers.dense(inputs=dense3, units=self.classes)

        return logits 