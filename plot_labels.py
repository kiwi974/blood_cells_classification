"""
    Script to plot classes distribution and one representative image per class.
"""

import os 
import skimage
import matplotlib.pyplot as plt
from itertools import groupby

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for (i,d) in enumerate(directories):
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".jpeg")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(i)
    return images, labels

ROOT_PATH = "./DATA/images"
train_data_directory = os.path.join(ROOT_PATH, "TRAIN")

images, labels = load_data(train_data_directory)

# Make a histogram with 62 bins of the `labels` data
labels_frequency = [len(list(group)) for key, group in groupby(labels)]
fig, ax = plt.subplots(figsize=(8,8))
x = [1,2,3,4]
classes_names = ['NEUTROPHIL', 'MONOCYTE', 'EOSINOPHILE', 'LYMPHOCYTE']
ax.bar(x, labels_frequency, color=['purple', 'steelblue', 'mediumseagreen', 'orange'], tick_label=classes_names)
ax.set_ylim((2450,2500))
ax.grid(True)
ax.set_title('Frequency distribution of each class in the training dataset', fontsize=16)
fig.savefig('images/labels_frequencies.png', bbox_inches='tight')
plt.show()


# Fill out the subplots with random images for each of the four classes
plt.figure(figsize=(15,15))
for label in range(0,4):
    image = images[labels.index(label)]
    plt.subplot(2,2, label+1)
    plt.axis('off')
    plt.title("{0} ({1})".format(classes_names[label], labels.count(label)), fontweight='bold')
    plt.imshow(image)
plt.show()
plt.savefig('images/class_representation.png', bbox_inches='tight')
