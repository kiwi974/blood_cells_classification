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

    string_labels = {'NEUTROPHIL':0, 'MONOCYTE':1, 'EOSINOPHIL':2, 'LYMPHOCYTE':3}



    def __init__(self, class_contingent):
        """
        Build a data loader. 
        Parameters : 
            - class_contingent : number of samples to load per class
        """
        self.class_contingent = class_contingent


    def fromPictToBinary(self,inputPath, outputPath):
        """
        Record in a binary format all pictures in a directory designed by inputPath, and record them
        in the directory designed by outputPath. All pictures are saved into the same binary file,
        previously stored in a numpy array. 
        Parameters : 
            - inputPath : path to folder containing the images to load, process and store in a binary format.
            - outputPath : path to the folder into which save the binary data.
        """
        # Make a list of the subdirectories in the directory designed by 'inputPath'
        directories = [d for d in os.listdir(inputPath) 
                    if os.path.isdir(os.path.join(inputPath, d))]
        labels = []
        images = []
        data = np.empty(shape=(0,60,80))
        classes = np.array([])
        for d in directories:
            print(d)
            label_directory = os.path.join(inputPath, d)
            file_names = [os.path.join(label_directory, f) 
                        for f in os.listdir(label_directory) 
                        if f.endswith(".jpeg")]
            # Add all the images in the subdirectory d to the data, and record the corresponding label 
            for (i,f) in enumerate(file_names):
                images.append(skimage.data.imread(f))
                labels.append(DataLoader.string_labels[d])
                if (i>=self.class_contingent):
                    break
            # Resize and grayscale the images 
            resized_images = [skimage.transform.resize(image, (60, 80), mode='constant') for image in images] 
            images = np.array(resized_images)

            # Convert images and labels into numpy arrays
            images = rgb2gray(images)
            labels = np.array(labels)

            # Concatenate data and free useless memory
            data = np.concatenate((data, images))
            classes = np.concatenate((classes, labels))
            del images
            del labels
            images = []
            labels = []

        # Record data and labels
        np.save(os.path.join(outputPath,'data.npy'), data, allow_pickle=False, fix_imports=False)
        np.save(os.path.join(outputPath,'labels.npy'), classes, allow_pickle=False, fix_imports=False)
         