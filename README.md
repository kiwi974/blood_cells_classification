# Blood Cells Classification 

An attempt to build and train a model able to perform a classification on blood cells. The study conducted here is practical. Studied aspects of convolutional neural networks are detailed in the section "The Study". 

# The Data 

Original data can be found here : https://www.kaggle.com/paultimothymooney/blood-cells. Data are images of 4 different types of blood cells : NEUTROPHIL, MONOCYTE, EOSINOPHILE, LYMPHOCYTE. Their frequencies distributions as well as their aspect are shown on the two following figures. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/33846463/54499553-72c17c80-4913-11e9-8afd-88db67dac0fc.png" width="600" 
</p>

<p align="center">
  <img src=https://user-images.githubusercontent.com/33846463/54499554-7a812100-4913-11e9-9d5b-31063404d06f.png>
</p>


# The branches

The difference between the two branches mainly lies in the way memory is managed. On the master, data are loaded and pre-processed at runtime. On the other side, on the branch 'placeholder_version', data have been pre-processed and saved under a numpy binary format. This binary data are read at runtime. It speeds up the data loading, but it still requires a huge amount of memory. With the placeholder version, less than a half of the data are used, in grayscale, with a poor quality because of image resizing. The use of the TensorFlow Dataset framework, on the master branch, allows to load all the images, in color, with a great quality, and without booking all the remaining memory on my GPU. 

# The Study 

As explained above, this study is practical, and is supposed to allow to become more familiar with some deep learning notions. It mainly revolves around 5 questions : 
  - Can the model be simplified and achieve comparable performances ? 
  - What is the influence of the learning rate on both training and testing performances ? 
  - What is the influence of pooling on both training and testing performances ? 
  - What is the influence of dropping on both training and testing performances ? 
  - What is the influence of data augmentation on both training and testing performances ? 
  
  
# The Default Model 

The default model which will be taken as reference for this study is given by its tensorflow representation. It achieves 100% of accuracy on training, while reaching about 65% on testing for grayscaled images, and 90% for colored images.   

(image below not up-to-date)
![reference_nn](https://user-images.githubusercontent.com/33846463/54516629-c5288a80-495f-11e9-97b1-573b3284b097.png)

In grayscale, we can notice that this model learns 'too much'. It probably 'memorizes' the images it sees, and do not generalize enough. This could be a first point on which to work for the following. 

Below the training and validation curves, as weel as loss functions, for the default model trained on both grayscaled and colored images. 

### For colored images

<p align="center">
<img src="https://user-images.githubusercontent.com/33846463/55686097-93528480-595d-11e9-9c75-f76200b88f66.png" width="420" height="420"> <img src="https://user-images.githubusercontent.com/33846463/55686085-7a49d380-595d-11e9-9e99-c42891627e37.png" width="420" height="420">
</p>

### For grayscaled images

<p align="center">
<img src="https://user-images.githubusercontent.com/33846463/55899524-47097d80-5bc5-11e9-85cb-5cc207bd8586.png" width="420" height="420"> <img src="https://user-images.githubusercontent.com/33846463/55899521-4670e700-5bc5-11e9-8c9d-0924fe1d8d28.png" width="420" height="420"> 
</p>

### Best model on grayscaled image

During previous researches, the best model found on grayscaled image reached the performances dipslayed by the following two graphs. 

<p align="center">
<img src="https://user-images.githubusercontent.com/33846463/55899603-89cb5580-5bc5-11e9-8ca1-1166b5878356.png" width="420" height="420"> <img src="https://user-images.githubusercontent.com/33846463/55899601-89cb5580-5bc5-11e9-88c9-555695bfe0f1.png" width="420" height="420"> 
</p>

For information purposes, I also trained and evaluated this model with colored images. As one can see, perfomances barely reach the performances of the reference model. Knowing that this model is more complex than the reference one, it seems that it is not interesting to continue with it. 

<p align="center">
<img src="https://user-images.githubusercontent.com/33846463/55907783-fcddc780-5bd7-11e9-9edd-aadf842a893e.png" width="420" height="420"> <img src="https://user-images.githubusercontent.com/33846463/55907781-fc453100-5bd7-11e9-936b-b0bfa6dfa600.png" width="420" height="420"> 
</p>

# Image Representation by the Default Autoencoder

On the following figure, one can preview how the autoencoder used tranforms the input images (update coming soon with the code). 

# The Equipment 

### Software : 
Tensorflow GPU is used, with CUDA 9.0 and cuDNN 7.0.5. 

### Hardware : 
Lenovo LEGION y530 with a Nvidia GeForce GTX 1060 GPU. With the default model presented above, around 1800 MiB of VRAM memory are required for grayscaled images, and around 3400 MiB for colored images. 
