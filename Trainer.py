"""
    Class which allows to train the neural network built by the ModelBuilder. Accuracy is the rate 
    of well classified samples. 
"""

import DataLoader, ModelBuilder
import tensorflow as tf
import numpy as np 
from sklearn.metrics import accuracy_score
import sys 


class Trainer:


    def __init__(self, input_placeholder, output_placeholder, train_size, test_size, logits, iterations,  train_data, train_labels, test_data, test_labels):
        """
        Build a trainer 
        Parameters : 
            - input_placeholder : the input placeholder of the network 
            - output_placeholder : the output placeholder of the network 
            - train_size : number of training samples per class 
            - test_size : number of testing samples per class 
            - output : output tensor of the network 
            - iterations : maximum number of iterations done for training 
            - train_data, train_labels : training samples (features and labels)
            - test_data, test_labels : testing samples (features and labels) 
        """

        self.input_placeholder = input_placeholder
        self.output_placeholder = output_placeholder
        self.train_size = train_size
        self.test_size = test_size
        self.output = logits
        self.iterations = iterations
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels



    
    def train(self, learning_rate, batch_size):
        """
        Train the model. Optimizer is Adam, loss is the sparse softmax cross entropy with logits, et predictions are 
        checked with argmax on logits. 
        Parameters : 
            - learning_rate : learning rate used by the optimizer 
            - batch_size : size of the training batches 
        """

        # Define a loss function
        with tf.variable_scope('loss'):
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.output_placeholder, 
                                                                                logits = self.output))
        # Define an optimizer 
        with tf.variable_scope('train_op'):
                train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        # Convert logits to label indexes
        with tf.variable_scope('eval'):
                correct_pred = tf.argmax(self.output, axis=1)

        # Record the different training and testing figures 
        losses, accuracies_it, train_accuracies, test_accuracies = [], [], [], []

        ##### Create and run a session #####

        tf.set_random_seed(1234)

        with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())

                for i in range(self.iterations+1):
                        batch_indexes = np.random.randint(4*self.train_size,size=batch_size)
                        _, loss_val = sess.run([train_op, loss], feed_dict={self.input_placeholder: self.train_data[batch_indexes], 
                                                                            self.output_placeholder: self.train_labels[batch_indexes]})
                        losses.append(loss_val)
                        if i % 50 == 0:       
                                #print("ITERATION : {0} ; Loss: {1}".format(i,loss_val))
                                arrow_length = int(10*(i/self.iterations))
                                progress_percent = (1000*(i/self.iterations))/10
                                sys.stdout.write('\r    \_ ITERATION : {1} / {2} ; loss = {3} [{4}>{5}{6}%]'.format(1, i, 
                                                 self.iterations, loss_val, '='*arrow_length,' '*(9-arrow_length), progress_percent))
                                sys.stdout.flush()
                                accuracies_it.append(i)
                                # Accuracy on training set 
                                acc_aux = correct_pred.eval(feed_dict={self.input_placeholder : self.train_data})
                                accuracy = accuracy_score(self.train_labels, acc_aux)
                                train_accuracies.append(accuracy)
                                # Accuracy on testing set 
                                acc_aux = correct_pred.eval(feed_dict={self.input_placeholder : self.test_data})
                                accuracy = accuracy_score(self.test_labels, acc_aux)
                                test_accuracies.append(accuracy)


                ##### Testing #####

                # Run predictions against the full test set.
                predicted = sess.run([correct_pred], feed_dict={self.input_placeholder: self.test_data})[0]

                # Calculate correct matches 
                match_count = sum([int(y == y_) for y, y_ in zip(self.test_labels, predicted)])

                # Calculate the accuracy
                accuracy = match_count / len(self.test_labels)

        return losses, accuracy, accuracies_it, train_accuracies, test_accuracies