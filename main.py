import DataLoaderTf, ModelBuilder, TrainerOpt
import tensorflow as tf
import sys 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.metrics import accuracy_score
import time 





# Load training and eval data

dlt = DataLoaderTf.DataLoaderTf()

batch_size = 100
train_dataset, training_count = dlt.load(batch_size)
test_dataset, testing_count = dlt.load(batch_size, pathToData='./DATA/images/TEST/')

# Create general iterator 
iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
next_element = iterator.get_next()


# Create the model 

should_drop = tf.placeholder(tf.bool)  # The actual value can be set up in 'Trainer.py' directly
dropout_rate1_placeholder = tf.placeholder(tf.float32, shape=(), name='dropout1_rate')

mb = ModelBuilder.ModelBuilder(120, 160, 4, dropout_rate1_placeholder, should_drop)

with tf.variable_scope('model') as scope: 
    logits = mb.networkBuilder(next_element[0])

trainer = TrainerOpt.TrainerOpt(logits, should_drop, dropout_rate1_placeholder, iterator, next_element, train_dataset, test_dataset)

# Define hyper parameters and others 
learning_rate = 0.0005 #Best found : 0.0005
epochs = 600
iterations = epochs*round(float(training_count)/batch_size)
drop1 = False
dropout_rate1 = 0.3
backup_folder = "models/"+"reference_model"    # folder in which save the model and other useful information 

start_time = time.time()
losses, accuracies_it, train_accuracies, test_accuracy = trainer.train(learning_rate, iterations, dropout_rate1, backup_folder, drop1)
end_time= time.time()

train_acc = round(train_accuracies[-1]*100,2)
test_accuracy = round(test_accuracy, 2)
print('\nComputation time is : {0}'.format('{0:.2f}'.format(float(end_time-start_time)/100.)))

# Save the hyperparameters in a file
hyperp_names = ",".join(['learning_rate', 'batch_size', 'nb_epochs', 'total_nb_iterations', 'should_drop', 'dropout_rate1', 'loss', 'train_accuracy', 'test_accuracy'])
hyperp = ",".join([str(learning_rate), str(batch_size), str(epochs), str(iterations), str(drop1), str(dropout_rate1), str(losses[-1]), str('{0}%'.format(train_acc)), 
                   str('{0}%'.format(test_accuracy))])

with open(backup_folder+'/hyper_parameters.txt', 'w+') as f:
    f.write(hyperp_names + '\n' + hyperp)
    f.close()



# Influence of the rate of the first dropout layer 
do1_exp = False
if (do1_exp):
    dropout_rate1_range = [.1,.2,.3,.4,.5,.6,.7,.8,.9]
    dropout_rate1_accuracies = []
    for (i,do_rate1) in enumerate(dropout_rate1_range):
        print('\n\n\nTRAINING n°{0} :\n'.format(i))
        _, accuracy, _, _, _ = trainer.train(learning_rate, batch_size, iterations, do_rate1, drop1)
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
    lr_range = [1, 0.1, 0.01, 0.001, 0.0005, 0.0001]
    train_accuracies = []
    test_accuracies = []
    for (i,lr) in enumerate(lr_range):
        print('\n\n\nTRAINING n°{0} :\n'.format(i+1))
        _, _, train_acc, test_acc = trainer.train(lr, iterations, dropout_rate1, backup_folder, drop1)
        train_accuracies.append(int(train_acc[-1]*100))
        test_accuracies.append(test_acc)

    fig,ax = plt.subplots(figsize=(15,15))
    ax.semilogx(lr_range, train_accuracies, label = 'training_acc', marker='x')
    ax.plot(lr_range, test_accuracies, label = 'valid_acc', marker='x')
    ax.set_title('Blood Cells Recognition Accuracy (learning rate)', fontsize=26)
    ax.set_xlabel('Learning Rate',fontsize=22)
    ax.set_ylabel('Accuracy (%)', fontsize=22)
    ax.legend()
    fig.savefig('lr.png', bbox_inches='tight')
    plt.show()
    print('\n')


# Influence of the batch size
bs_exp = False
if (bs_exp):
    bs_range = [5, 20, 50, 100, 200]
    bs_accuracies = []
    for (i,bs) in enumerate(bs_range):
        print('\n\n\nTRAINING n°{0} :\n'.format(i+1))
        _, accuracy, _, _, _ = trainer.train(learning_rate, bs, iterations, dropout_rate1, drop1)
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
    print("\nTraining accuracy: {:.3f}%".format(train_acc))
    print("Testing accuracy: {:.3f}%\n".format(test_accuracy))

    fig,ax = plt.subplots(figsize=(15,15))
    ax.plot(np.arange(iterations+1),losses)
    ax.set_title('Blood Cells Recognition Loss (batch size : {0})'.format(batch_size), fontsize=26)
    ax.set_xlabel('Iterations',fontsize=22)
    ax.set_ylabel('Cross Entropy Loss', fontsize=22)
    fig.savefig(backup_folder+'/loss.png', bbox_inches='tight')
    plt.show()

    fig,ax = plt.subplots(figsize=(15,15))
    ax.plot(np.arange(iterations+1), train_accuracies, label = 'Accuracy on training set')
    #ax.plot(accuracies_it, test_accuracies, label = 'Accuracy on testing set')
    ax.set_title('Blood Cells Recognition Accuracy (batch size : {0}, lr = {1})'.format(batch_size, learning_rate), fontsize=26)
    ax.set_xlabel('Iterations',fontsize=22)
    ax.set_ylabel('Accuracy', fontsize=22)
    ax.legend()
    fig.savefig(backup_folder+'/training_accuracy.png', bbox_inches='tight')
    plt.show()
