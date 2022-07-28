# Overview

This is an image classification project by comparing shallow learning with Deep learning. The goal is to see which model ie SVM or Convolutional Neural Networks 
does a better job at classifying labelled images. The metric to determine performance is the accuracy of the model. Overall, the CNNs do a much better at 
classifying images compared to SVMs because CNNs use multiple layers and therefore able to extract features that a shallow learning model based on creating
a boundary cannot.

## EDA and Data Cleaning
 
For SVM, the images needed to be resized and flattened to arrays as it is a mathematical model that can only take vectors and numbers as input. The process
for flattening and resize involved creating functions and making use of numpy methods. I then reconstructed the images from the flattened arrays to confirm
that the flattening did not distort the images to maintain the integrity of training.

For CNNs, I took advantage of ImageDataGenerator from keras that allows real-time augmentation and rescaling of image data during training. This means that
unlike for SVM, it is easy to determine what shape and size you want the images to be without doing the flattening and resize yourself. Instead, it is an
optimized function that does all the preprocessing of the images behind the scenes and makes sure that all the features are kept which leads to a richer
classification.

## SVM vs CNN
I trained three types of SVM: rbf, poly and linear. For each, I created a pipeline to ensure that I am not repeating common steps and that each model gets
data that has been processed in the same way. I also used Gridsearch to find the best combination of hyperparameters for optimal performance of the models.
The best model was the rbf kernel with an accuracy of 67%.

For CNN, I used a basic out of the box CNN and transfer learning with VGG16. As mentioned, I used ImageDataGenerator for image preprocessing, rescale and 
real-time augmentation as the model trained. The basic CNN had ok accuracy: 66% which is the same as the rbf kernel, and high loss. This was not impressive
especially because it took significantly longer to train than SVM. On the other hand, the model with transfer acheived a validation accuracy of 82% and had
low loss. I use dropout and early stopping to prevent overfitting in both the basic and tranfer learning models. Removing more of the Dense layers and 
leaving only prediction layer also significantly improved the model accuracy. Transfer learning achieved high accuracy within just 6 epochs as compared to
the basic CNN.
