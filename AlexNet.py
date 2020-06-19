#Importing packages
import tensorflow as tf
from skimage import data
import matplotlib.pyplot as plt
import os
import random
from skimage import transform
import skimage.data
import skimage.transform
from tensorflow.contrib.layers import flatten
import numpy as np


tf.logging.set_verbosity(tf.logging.INFO)

##The skeleton of our model
#Load images from folders to tables and build 2lists:images,labels.
def load_data(data_dir):
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".jpg")]
        for f in file_names:
            images.append(data.imread(f))
            labels.append(int(d))
    return images, labels


#This is our cnn model
def create_conv_model(input_data):

    #reshape and flatten tne input_data
    input_layer = tf.reshape(input_data, [-1,50,50,3])


    #1st Convolutional Layer and pooling layer
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters =96,
        kernel_size =[11,11],
        padding ="same",
        activation = tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides =4)

    #2nd Convolutional Layer and pooling layer
    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters =128,
        kernel_size =[5,5],
        padding ="same",
        activation = tf.nn.relu
        )

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides =4)

    #3rd Convolutional Layer and pooling layer
    conv3 = tf.layers.conv2d(
        inputs = input_layer,
        filters =384,
        kernel_size =[3,3],
        padding ="same",
        activation = tf.nn.relu)

    pool3 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides =4)

    #4th Convolutional Layer and pooling layer
    conv4 = tf.layers.conv2d(
        inputs = input_layer,
        filters =192,
        kernel_size =[3,3],
        padding ="same",
        activation = tf.nn.relu)

    pool4 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides =4)

    #5th Convolutional Layer and pooling layer
    conv5 = tf.layers.conv2d(
        inputs = input_layer,
        filters =128,
        kernel_size =[3,3],
        padding ="same",
        activation = tf.nn.relu)

    pool5 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides =4)

    #Flatten layer
    pool5_flat = flatten(pool5)

    #fully connected layer
    logit1 = tf.layers.dense(inputs = pool5_flat, units =1000)
    logit2 = tf.layers.dense(inputs = logit1, units =1000)
    logit = tf.layers.dense(inputs = logit2, units = 3)

    return logit


#Declare  x and y as placeholders
x = tf.placeholder(dtype = tf.float32, shape =[None, 50, 50, 3])
y = tf.placeholder(dtype = tf.int32, shape = [None])

#Call the model function
logits = create_conv_model(x)

#Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))

#define an optimizer "Adam optimizer algrithm"
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

#Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

#Define an accuracy metric
#accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))



##The Data:blood in the human body

tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

#Load the test data and training data
train_data_dir= "C:/Users/Lenovo/Desktop/Training"
images, labels = load_data(train_data_dir)


#Resizing images
images50 = [transform.resize(image, (50, 50)) for image in images]



list_loss=[]

#Training with 20 EPOCH
for i in range(20):
    print('EPOCH', i)
    _, loss_val = sess.run([train_op, loss], feed_dict={x: images50, y: labels})
    list_loss.append(loss_val)
    print("Loss: ", loss_val)
    print('DONE WITH EPOCH')


#Test data
test_data_dir ="C:/Users/Lenovo/Desktop/Testing"
test_images, labels_test = load_data(test_data_dir)

#Import 'skimage'
from skimage import transform

#Transform the images to 50 by 50 pixels
test_images50 = [transform.resize(image, (50,50)) for image in test_images]

#Run predictions against the full test set
predicted = sess.run([correct_pred], feed_dict={x: test_images50})[0]

#Calculate correct matches
match_count = sum([int(y == y_) for y, y_ in zip(labels_test, predicted)])

#Calculate the accuracy
accuracy = match_count / len(labels_test)

#Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))



#Evaluation

import random
#Pick 10 random images
sample_indexes = random.sample(range(len(images50)), 10)
sample_images = [images50[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

#Run the "correct_pred" operation
predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]

#Print the real and predicted labels
print(sample_labels)
print(predicted)

#Display the predictions and the ground truth visually
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2, 1+i)
    plt.axis('off')
    color = 'green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:    {0}\n Prediction:    {1}".format(truth, prediction), fontsize=12, color=color)
    plt.imshow(sample_images[i], cmap="gray")
plt.show()


#Display the graph: epoch according to loss function
x_loss=[]
    
x_loss=[i for i in range(20)]
plt.plot(x_loss,list_loss)
plt.show()






