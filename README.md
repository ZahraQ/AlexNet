# AlexNet

Neural Networks are essentially mathematical models to solve an optimization problem. They are
made of neurons, the basic computation unit of neural networks. A neuron takes an input(say x), do some
computation on it(say: multiply it with a variable w and adds another variable b ) to produce a value (say;
z= wx+b). This value is passed to a non-linear function called activation function(f) to produce the final
output(activation) of a neuron.

The Alexnet CNN model is trained on a subset of the ImageNet database , which is used in 
ImageNet Large-Scale Visual Recognition Challenge (ILSVRC). The model is
trained on more than a million images and can classify images into 1000 object categories. As a result, the
model has learned rich feature representations for a wide range of images.

And as a Dataset , we used Labeled Faces in the Wild, a database of face photographs designed for
studying the problem of unconstrained face recognition. The data set contains more than 13,000 images of
faces collected from the web. Each face has been labeled with the name of the person
pictured. Information about the dataset:

· 13233 images

· 5749 people

· 1680 people with two or more images

## Background knowledge

### AlexNet CNN model

This architecture was one of the first deep networks to push ImageNet Classification accuracy by a
significant stride in comparison to traditional methodologies. It is composed of 5 convolutional layers
followed by 3 fully connected layers :

![AlexNet](https://user-images.githubusercontent.com/38895618/85140121-baccfe80-b23c-11ea-96fb-e6b379f28e60.PNG)

AlexNet, proposed by Alex Krizhevsky, uses ReLu(Rectified Linear Unit) for the non-linear part, instead
of a Tanh or Sigmoid function which was the earlier standard for traditional neural networks. ReLu is given
by f(x) = max(0,x)

### Task and problem

The aim of this lab is to build an application in computer vision using « Alexnet CNN model » and
« Labeled Faces in the Wild » dataset , then to visualize the accuracy obtained and display the graph model
made by tensorflow .

### Results and graph

Displaying EPOCH , loss function and accuracy. The accuracy does not only depend on the network but also
on the amount of data available for training and the number of accuracy.
