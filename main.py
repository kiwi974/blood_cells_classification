import DataLoader, ModelBuilder, Trainer
import tensorflow as tf
import sys 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import accuracy_score



# Load training and eval data

train_data, train_labels = np.load('DATA/binaries/TRAIN/data.npy'), np.load('DATA/binaries/TRAIN/labels.npy')
test_data, test_labels = np.load('DATA/binaries/TEST/data.npy'), np.load('DATA/binaries/TEST/labels.npy')

train_size = train_data.shape[0]


# Create the model 

x = tf.placeholder(dtype = tf.float32, shape = [None, 60, 80])
y = tf.placeholder(dtype = tf.int32, shape = [None])
should_drop = tf.placeholder(tf.bool)  # The actual value can be set up in 'Trainer.py' directly
dropout_rate1_placeholder = tf.placeholder(tf.float32, shape=(), name='dropout1_rate')

mb = ModelBuilder.ModelBuilder(60, 80, 4 ,x , dropout_rate1_placeholder, should_drop)

with tf.variable_scope('model') as scope: 
        logits = mb.networkBuilder()

trainer = Trainer.Trainer(x,y,train_size, logits, should_drop, dropout_rate1_placeholder, train_data, train_labels, test_data, test_labels)


learning_rate = 0.000092
batch_size = 100
max_iterations = 15000
dropout_rate1 = 0.2

losses, accuracy, accuracies_it, train_accuracies, test_accuracies = trainer.train(learning_rate, batch_size, max_iterations, dropout_rate1)


# Influence of the rate of the first dropout layer 
do1_exp = False
if (do1_exp):
    dropout_rate1_range = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
    dropout_rate1_accuracies = []
    for (i,do_rate1) in enumerate(dropout_rate1_range):
        print('\n\n\nTRAINING n°{0} :\n'.format(i))
        _, accuracy, _, _, _ = trainer.train(learning_rate, batch_size, max_iterations, do_rate1)
        dropout_rate1_accuracies.append(accuracy)

    fig,ax = plt.subplots(figsize=(15,15))
    ax.plot(dropout_rate1_range,dropout_rate1_accuracies)
    ax.set_title('Blood Cells Recognition Accuracy (dropout rate 1)', fontsize=26)
    ax.set_xlabel('Dropout Rate 1',fontsize=22)
    ax.set_ylabel('Inference Accuracy', fontsize=22)
    fig.savefig('dropout_rate1.png', bbox_inches='tight')
    plt.show()
    print('\n')



# Influence of the learning rate 
lr_exp = False
if (lr_exp):
    lr_range = [1, 0.1, 0.01, 0.001, 0.0001]
    lr_accuracies = []
    for (i,lr) in enumerate(lr_range):
        print('\n\n\nTRAINING n°{0} :\n'.format(i+1))
        _, accuracy, _, _, _ = trainer.train(lr, batch_size, max_iterations, dropout_rate1)
        lr_accuracies.append(accuracy)

    fig,ax = plt.subplots(figsize=(15,15))
    ax.plot(lr_range,lr_accuracies)
    ax.set_title('Blood Cells Recognition Accuracy (learning rate)', fontsize=26)
    ax.set_xlabel('Learning Rate',fontsize=22)
    ax.set_ylabel('Inference Accuracy', fontsize=22)
    fig.savefig('lr2.png', bbox_inches='tight')
    plt.show()
    print('\n')


# Influence of the batch size
bs_exp = False
if (bs_exp):
    bs_range = [5, 20, 50, 100, 200]
    bs_accuracies = []
    for (i,bs) in enumerate(bs_range):
        print('\n\n\nTRAINING n°{0} :\n'.format(i+1))
        _, accuracy, _, _, _ = trainer.train(learning_rate, bs, max_iterations, dropout_rate1)
        bs_accuracies.append(accuracy)

    fig,ax = plt.subplots(figsize=(15,15))
    ax.plot(bs_range,bs_accuracies)
    ax.set_title('Blood Cells Recognition Accuracy (batch_size)', fontsize=26)
    ax.set_xlabel('Batch Size',fontsize=22)
    ax.set_ylabel('Inference Accuracy', fontsize=22)
    fig.savefig('bs.png', bbox_inches='tight')
    plt.show()
    print('\n')


if (True):
    print("\nAccuracy: {:.3f}".format(accuracy))

    fig,ax = plt.subplots(figsize=(15,15))
    ax.plot(np.arange(max_iterations+1),losses)
    ax.set_title('Blood Cells Recognition Loss (batch size : {0})'.format(batch_size), fontsize=26)
    ax.set_xlabel('Iterations',fontsize=22)
    ax.set_ylabel('Cross Entropy Loss', fontsize=22)
    fig.savefig('loss.png', bbox_inches='tight')
    plt.show()

    fig,ax = plt.subplots(figsize=(15,15))
    ax.plot(accuracies_it, train_accuracies, label = 'Accuracy on training set')
    ax.plot(accuracies_it, test_accuracies, label = 'Accuracy on testing set')
    ax.set_title('Blood Cells Recognition Accuracy (batch size : {0}, lr = {1})'.format(batch_size, learning_rate), fontsize=26)
    ax.set_xlabel('Iterations',fontsize=22)
    ax.set_ylabel('Accuracy', fontsize=22)
    ax.legend()
    fig.savefig('accuracy.png', bbox_inches='tight')
    plt.show()
