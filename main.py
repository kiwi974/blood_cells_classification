import DataLoader, ModelBuilder, Trainer
import tensorflow as tf
import sys 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import accuracy_score



# Load training and eval data

train_size = 700
dl = DataLoader.DataLoader(train_size)
train_data, train_labels = dl.preprocess('training')

test_size = 300
dl_test = DataLoader.DataLoader(test_size)
test_data, test_labels = dl_test.preprocess('testing')

learning_rate = 0.0002
batch_size = 50
max_iterations = 3000
dropout_rate1 = 0.7


# Create the model 

x = tf.placeholder(dtype = tf.float32, shape = [None, 60, 80])
y = tf.placeholder(dtype = tf.int32, shape = [None])
should_drop = tf.placeholder(tf.bool)
dropout_rate1_placeholder = tf.placeholder(tf.float32, shape=(), name='dropout1_rate')

mb = ModelBuilder.ModelBuilder(60,80,4,x, dropout_rate1_placeholder, should_drop)

with tf.variable_scope('model') as scope: 
        logits = mb.networkBuilder()

trainer = Trainer.Trainer(x,y,train_size, test_size, logits, should_drop, dropout_rate1_placeholder, train_data, train_labels, test_data, test_labels)

losses, accuracy, accuracies_it, train_accuracies, test_accuracies = trainer.train(learning_rate, batch_size, max_iterations, dropout_rate1)



# Print the accuracy
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
