""" 
    Class which allows to load the images in memory and preprocess them. 
"""

import os
from skimage import transform
import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import tensorflow as tf
import random
import sys

class DataLoader:

    ROOT_PATH = "DATA/images"
    train_data_directory = os.path.join(ROOT_PATH, "TRAIN")
    test_data_directory = os.path.join(ROOT_PATH, "TEST")

    string_labels = {'NEUTROPHIL':0, 'MONOCYTE':1, 'EOSINOPHIL':2, 'LYMPHOCYTE':3}



    def __init__(self, class_contingent):
        """
        Build a data loader. 
        Parameters : 
            - class_contingent : number of samples to load per class
        """
        self.class_contingent = class_contingent

    
    def load_data(self,data_directory, string_labels):
        """
        Load the data into memory. 
        Parameters : 
            - data_directory : directory containing all the pictures
            - string_labels : dictionnary associated each string label to an int label 
        Returns : 
            - images : an array containing the loaded images 
            - labels : an array containing the corresponding labels 
        """
        # Make a list of the subdirectories in the directory 'data_directory'
        directories = [d for d in os.listdir(data_directory) 
                    if os.path.isdir(os.path.join(data_directory, d))]
        # Initialize the which will contain the labels and the data (images under ppm format)
        labels = []
        images = []
        for d in directories:
            print(d)
            label_directory = os.path.join(data_directory, d)
            file_names = [os.path.join(label_directory, f) 
                        for f in os.listdir(label_directory) 
                        if f.endswith(".jpeg")]
            # Add all the images in the subdirectory d to the data, and keep the corresponding label 
            for (i,f) in enumerate(file_names):
                images.append(skimage.data.imread(f))
                labels.append(string_labels[d])
                if (i>=self.class_contingent):
                    break
        return images, labels

    
    def preprocess(self, mode):
        """
            Preprocess the images, by rescaling and grayscaling. 
            Parameters: 
                - mode : 'training' or 'testing', according if we are loading training or testing data
            Remark : 
                the main reasons of the preprocessing is the reduce the memory consumption of the programm. 
        """
        if (mode == 'training'):
            images, labels = self.load_data(DataLoader.train_data_directory, DataLoader.string_labels)
        else:
            images, labels = self.load_data(DataLoader.test_data_directory, DataLoader.string_labels)

        resized_images = [skimage.transform.resize(image, (60, 80), mode='constant') for image in images] 
        images = np.array(resized_images)
        images = rgb2gray(images)

        return images, np.array(labels)