"""
    Class which allows to train the neural network built by the ModelBuilder. Accuracy is the rate 
    of well classified samples. 
"""

import DataLoaderTf, ModelBuilder
import tensorflow as tf
import numpy as np 
from sklearn.metrics import accuracy_score
import sys 


class TrainerOpt:


    def __init__(self, logits, should_drop, dropout_rate1_placeholder, iterator, next_element, train_dataset, test_dataset):
        """
        Build a trainer 
        Parameters : 
            - train_size : total number of training samples
            - output : output tensor of the network 
            - should_drop : boolean the apply or not dropout layer 1 (depending on training or inference mode)
            - dropout_rate1_placeholder : dropout rate of the first dropout layer
            - train_data, train_labels : training samples (features and labels)
            - test_data, test_labels : testing samples (features and labels) 
        """
        self.logits = logits
        self.should_drop = should_drop
        self.dropout_rate1_placeholder = dropout_rate1_placeholder
        self.iterator = iterator
        self.next_element = next_element
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
    
    def train(self, learning_rate, iterations, do_rate1, backup_folder):
        """
        Train the model. Optimizer is Adam, loss is the sparse softmax cross entropy with logits, et predictions are 
        checked with argmax on logits. 
        Parameters : 
            - learning_rate : learning rate used by the optimizer 
            - batch_size : size of the training batches 
            - max_iterations : maximum number of iterations done for training 
            - do_rate1 : dropout rate of the first dropout layer
            - backu_folder : folder to which save the current trained model
        """

        # Make datasets that we can initialize separately, but using the same structure via the common iterator
        training_init_op = self.iterator.make_initializer(self.train_dataset)
        testing_init_op = self.iterator.make_initializer(self.test_dataset)

        # Define a loss function
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.next_element[1], 
                                                                                logits = self.logits))
        # Define an optimizer 
        with tf.variable_scope('train_op', reuse=tf.AUTO_REUSE):
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        # Convert logits to label indexes
        with tf.variable_scope('correct_pred', reuse=tf.AUTO_REUSE):
            correct_pred = tf.argmax(self.logits, axis=1)

        # Equality between prediction and target prediciton 
        with tf.variable_scope('equality', reuse=tf.AUTO_REUSE):
            equality = tf.equal(correct_pred, tf.argmax(self.next_element[1],axis=1)) #tf.argmax(self.current_labels, 0))

        # Accuracy of the prediction 
        with tf.variable_scope('accuracy', reuse=tf.AUTO_REUSE):
            accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Record the different training and testing figures 
        losses, accuracies_it, train_accuracies, test_accuracies = [], [], [], []

        ##### Create and run a session #####

        tf.set_random_seed(1234)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True


        with tf.Session(config=config) as sess:

                writer = tf.summary.FileWriter("output", sess.graph)

                sess.run(tf.global_variables_initializer())


                for i in range(iterations):

                    if (i==0):
                        print('\n\nTRAINING & TESTING : ')

                    sess.run(training_init_op)

                    _, loss_val = sess.run([train_op, loss], feed_dict={self.should_drop : False,
                                                                        self.dropout_rate1_placeholder : do_rate1})
                    losses.append(loss_val)

                    if (i%20 == 0):
                        accuracies_it.append(i)

                        # Accuracy on training set 
                        prediction, train_acc = sess.run([correct_pred, accuracy], feed_dict={self.should_drop : False,
                                                                    self.dropout_rate1_placeholder : do_rate1})

                        train_accuracies.append(train_acc) 

                        # Accuracy on testing set 
                        #if (i%100 == 0):
                        sess.run(testing_init_op)
                        test_acc = sess.run([accuracy], feed_dict={self.should_drop : False,
                                                                        self.dropout_rate1_placeholder : do_rate1})
                        test_accuracies.append(test_acc[0])

                        # Display the results 
                        arrow_length = int(10*(i/iterations))
                        progress_percent = (int(1000*(i/iterations)))/10
                        sys.stdout.write('\r    \_ EPOCH : {0} ; ITERATION : {1} / {2} ; loss = {3} ; training_acc = {4} ; testing_acc = {5} [{6}>{7}{8}%]'.format(
                                                 i//100+1, i, iterations, '{0:.6f}'.format(loss_val), '{0:.6f}'.format(train_acc), '{0:.6f}'.format(test_acc[0]),
                                                 '='*arrow_length,' '*(9-arrow_length), progress_percent))
                        sys.stdout.flush()
                
                
                save_path = saver.save(sess, "{0}/model.ckpt".format(backup_folder))
                print('\n\nModel saved in %s' % save_path)
                                


                writer.close()

        return losses, accuracies_it, train_accuracies, test_accuracies