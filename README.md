# Blood Cells Classification 

An attempt to build and train a model able to perform a classification on blood cells. The study conducted here is practical. Studied aspects of convolutional neural networks are detailed in the section "The Study". 

# The Data 

Original data can be found here : https://www.kaggle.com/paultimothymooney/blood-cells. Data are images of 4 different types of blood cells : NEUTROPHIL, MONOCYTE, EOSINOPHILE, LYMPHOCYTE. Their frequencies distributions as well as their aspect are shown on the two following figures. 


![labels_frequencies](https://user-images.githubusercontent.com/33846463/54499553-72c17c80-4913-11e9-8afd-88db67dac0fc.png)


![classes](https://user-images.githubusercontent.com/33846463/54499554-7a812100-4913-11e9-9d5b-31063404d06f.png)

# The Study 

As explained above, this study is practical, and is supposed to allow to become more familiar with some deep learning notions. It mainly revolves around 5 questions : 
  - Can the model be simplified and achieve comparable performances ? 
  - What is the influence of the learning rate on both training and testing performances ? 
  - What is the influence of pooling on both training and testing performances ? 
  - What is the influence of dropping on both training and testing performances ? 
  - What is the influence of data augmentation on both training and testing performances ? 
  
  
# The Default Model 

The default model which will be taken as reference for this study is given by its tensorflow representation. It achieves 100% of accuracy on training, while reaching about 65% on testing.   

![reference_nn](https://user-images.githubusercontent.com/33846463/54516629-c5288a80-495f-11e9-97b1-573b3284b097.png)

First, we can notice that this model learns 'too much'. It probably 'memorizes' the images it sees, and do not generalize that much. This could be a first point on which to work for the following. 


# Equipment 

### Software : 
Tensorflow GPU is used, with CUDA 9.0 and cuDNN 7.0.5. 

### Hardware : 
Lenovo LEGION y530 with Nvidia GeForce GTX 1060 GPU. With the default model presented above, 1829 MiB of VRAM memory are required. 
