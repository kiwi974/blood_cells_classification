"""
    Class which allows to build a neural network, with its different training and 
    testing parameters. 
    The built network must be fed with grayscaled images. 
"""

import tensorflow as tf

class ModelBuilder:


    def __init__(self, input_height, input_width, classes, input_data, mode=True):
        """
        Build a ModelBuilder. 
        Parameters : 
            - input_height : height of the input images 
            - input_width : width of the input images 
            - classes : number of classes of the model 
            - input_data : the data fedding the model (a placeholder in practise)
            - mode : default is True for training, otherwise it is False for inference
        """
        self.input_height = input_height
        self.input_width = input_width
        self.classes = classes
        self.input_data = input_data
        self.mode = mode 



    

    def networkBuilder(self):
        """     
        Build the architecture of the convolutional network. 
        """
        with tf.variable_scope('input_layer'):
            # Build the input layer, with a dynamically computed batch size
            input_layer = tf.reshape(self.input_data, [-1, self.input_height, self.input_width, 1])

        with tf.variable_scope('conv_1'):
            # Convolutional Layer 1 : apply 32 5x5 filters, with ReLU activation functions
            conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=32,
                kernel_size=[5,5],
                padding="same",
                activation=tf.nn.relu
            )

        with tf.variable_scope('pool_1'):
            # Pooling Layer 1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        with tf.variable_scope('norm_1'):
            # Normalization Layer 1
            norm1 = tf.nn.local_response_normalization(input=tf.to_float(pool1))

        with tf.variable_scope('conv_2'):
            # Convolutional Layer 2 : apply 64 5x5 filters
            conv2 = tf.layers.conv2d(
                inputs=norm1,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu
            )

        with tf.variable_scope('norm_2'):
            # Normalization Layer 2
            norm2 = tf.nn.local_response_normalization(input=tf.to_float(conv2))

        with tf.variable_scope('pool_2'):
            # Pooling Layer 2
            pool2 = tf.layers.max_pooling2d(inputs=norm2, pool_size=[2, 2], strides=2)

        with tf.variable_scope('pool2_flat'):
            # Entry Flatten Layer 
            pool2_flat = tf.reshape(pool2, [-1, 15 * 20 * 64]) 

        with tf.variable_scope('dense_1'):
            # Dense Layer 1
            dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        
        with tf.variable_scope('dropout_1'):
            # Dropout Layer 1
            dropout1 = tf.layers.dropout(inputs=dense1, rate=0.4, training=self.mode) 

        with tf.variable_scope('dense_2'):
            # Dense Layer 2
            dense2 = tf.layers.dense(inputs=dropout1, units=1024, activation=tf.nn.relu)
            
            # Dropout Layer 2
            #dropout2 = tf.layers.dropout(
            #    inputs=dense2, rate=0.4, training= (mode==tf.estimator.ModeKeys.TRAIN))

            # Logits Layer
        with tf.variable_scope('logits'):
            logits = tf.layers.dense(inputs=dense2, units=self.classes)

            return logits 