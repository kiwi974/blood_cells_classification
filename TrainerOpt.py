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


    def __init__(self, input_placeholder, output_placeholder, logits, should_drop, dropout_rate1_placeholder, iterator, next_element, train_dataset, test_dataset):
        """
        Build a trainer 
        Parameters : 
            - input_placeholder : the input placeholder of the network 
            - output_placeholder : the output placeholder of the network 
            - train_size : total number of training samples
            - output : output tensor of the network 
            - should_drop : boolean the apply or not dropout layer 1 (depending on training or inference mode)
            - dropout_rate1_placeholder : dropout rate of the first dropout layer
            - train_data, train_labels : training samples (features and labels)
            - test_data, test_labels : testing samples (features and labels) 
        """

        self.input_placeholder = input_placeholder
        self.output_placeholder = output_placeholder
        self.output = logits
        self.should_drop = should_drop
        self.dropout_rate1_placeholder = dropout_rate1_placeholder
        self.iterator = iterator
        self.next_element = next_element
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    
    def train(self, learning_rate, epochs, do_rate1, backup_folder):
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
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.next_element[1], 
                                                                                logits = self.output))
        # Define an optimizer 
        with tf.variable_scope('train_op', reuse=tf.AUTO_REUSE):
                train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        # Convert logits to label indexes
        with tf.variable_scope('eval', reuse=tf.AUTO_REUSE):
                correct_pred = tf.argmax(self.output, axis=1)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Record the different training and testing figures 
        losses, accuracies_it, train_accuracies, test_accuracies = [], [], [], []

        ##### Create and run a session #####

        tf.set_random_seed(1234)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        print('\n\n')

        with tf.Session(config=config) as sess:

                writer = tf.summary.FileWriter("output", sess.graph)

                sess.run(tf.global_variables_initializer())

                train_labels = self.next_element[1]

                for i in range(epochs):
                    sess.run(training_init_op)
                    _, loss_val, train_pred = sess.run([train_op, loss, correct_pred], feed_dict={self.input_placeholder: self.next_element[0].eval(), 
                                                                            self.output_placeholder: train_labels,
                                                                            self.should_drop : False,
                                                                            self.dropout_rate1_placeholder : do_rate1})
                    losses.append(loss_val)

                    if (i%5 == 0):
                        accuracies_it.append(i)

                        # Accuracy on training set 
                        train_acc = accuracy_score(train_labels, train_pred)
                        train_accuracies.append(train_acc)

                        # Accuracy on testing set 
                        sess.run(testing_init_op)
                        test_pred = correct_pred.eval(feed_dict={self.input_placeholder : self.next_element[0], 
                                                                       self.should_drop : False,
                                                                       self.dropout_rate1_placeholder : do_rate1})
                        test_acc = accuracy_score(self.next_element[1], test_pred)
                        test_accuracies.append(test_acc)

                        # Display the results 
                        arrow_length = int(10*(i/epochs))
                        progress_percent = (int(1000*(i/epochs)))/10
                        sys.stdout.write('\r    \_ ITERATION : {1} / {2} ; loss = {3} ; training_acc = {4} ; testing_acc = {5} [{6}>{7}{8}%]'.format(
                                                 1, i, epochs, '{0:.6f}'.format(loss_val), '{0:.6f}'.format(train_acc), '{0:.6f}'.format(test_acc),
                                                 '='*arrow_length,' '*(9-arrow_length), progress_percent))
                        sys.stdout.flush()
                
                
                save_path = saver.save(sess, "{0}/model.ckpt".format(backup_folder))
                print('\n\nModel saved in %s' % save_path)
                                


                ##### Testing #####

                # Run predictions against the full test set.
                #predicted = sess.run([correct_pred], feed_dict={self.input_placeholder: self.test_data, 
                #                                                self.should_drop : False, 
                #                                                 self.dropout_rate1_placeholder : do_rate1})[0]

                # Calculate correct matches 
                #match_count = sum([int(y == y_) for y, y_ in zip(self.test_labels, predicted)])

                # Calculate the accuracy
                #accuracy = match_count / len(self.test_labels)

                writer.close()

        return losses, accuracies_it, train_accuracies, test_accuracies