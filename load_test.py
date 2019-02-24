"""
    Script to save the data into a numpy binary format. 
"""

import DataLoader 
import numpy as np

# Number of training samples to keep per class 
training_samples_contingent = 700

# Number of testing samples to keep per class
testing_samples_contingent = 250

# Data loaders for training and testing data 
dl_train = DataLoader.DataLoader(training_samples_contingent)
dl_test = DataLoader.DataLoader(testing_samples_contingent)

# Handle with training and testing data 
dl_train.fromPictToBinary('DATA/images/TRAIN', 'DATA/binaries/TRAIN')
dl_train.fromPictToBinary('DATA/images/TEST', 'DATA/binaries/TEST')