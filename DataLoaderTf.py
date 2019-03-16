"""
  Class which allows to load data thanks to the TensorFlow Dataset framework 
  Sources : https://www.tensorflow.org/tutorials/load_data/images
            https://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import pathlib
import random 
import matplotlib.pyplot as plt
import IPython.display as display
import time

#tf.enable_eager_execution()

class DataLoaderTf:

  def __init__(self):
    self.name = 'DataLoader'


  ###### Load and format the images #####
  def preprocess_image(self, image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize_images(image, [120, 160])
    image /= 255.0  # normalize to [0,1] range
    return image

  ###### Load an image with path ######
  def load_and_preprocess_image(self, path):
    image = tf.read_file(path)
    return self.preprocess_image(image)



  def load(self, batch_size, pathToData='./DATA/images/TRAIN/'):

    ###### Retrieve the images ######
    data_root = pathlib.Path(pathToData)
    all_image_paths = list(data_root.glob('*/*'))   # Each element is of type 'pathlib.PosixPath'
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)

    image_count = len(all_image_paths)
    print("There are {0:.0f} training images in total." .format(image_count))

    ###### Determine the label of each image ######

    # List the available lables 
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

    # Assign an index to each label 
    label_to_index = dict((name, index) for index,name in enumerate(label_names))

    # Create a list of corresponding labels 
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

    print("First 10 labels indices: ", all_image_labels[:10])


    ##### Build a tf.data.Dataset #####

    # Slicing the array of strings results in a dataset of strings 
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

    # Load and format images on the fly bu mapping preprocess_image over the dataset of paths
    image_ds = path_ds.map(self.load_and_preprocess_image)

    # Create a dataset with labels 
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64)).map(lambda z: tf.one_hot(z,4))

    # Zip images and labels 
    ds = tf.data.Dataset.zip((image_ds, label_ds)).shuffle(500).repeat().batch(batch_size)

    return ds, image_count



