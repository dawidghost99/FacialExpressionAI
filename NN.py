
import tensorflow as tf
from tensorflow import keras
import numpy as np
import mnist
import matplotlib.pyplot as plt
import random
import cv2
import pandas as pd
import time
import os
import pickle
import sklearn
from tqdm import tqdm
import CreateDataSet as DS



#DS.expressions()

trainImages = pickle.load(open('train_images.pickle','rb'))
trainLabels = pickle.load(open('train_lables.pickle','rb'))


testImages = pickle.load(open('testing_images.pickle','rb'))
testLabels = pickle.load(open('testing_lables.pickle','rb'))
IMG_SIZE = 256

trainImages = np.array(trainImages).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
trainLabels = np.array(trainLabels)

testImages = np.array(testImages).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
testLabels = np.array(testLabels)

#print(trainImages, trainLabels)

end = 0.0
start = 0.0


print ("number of training images: ",len(trainImages))
print("number of testing images", len(testImages))

def NeuralNet(inputNeurons, numepochs,run):


    start = time.time()

    model = keras.Sequential([
    
    keras.layers.Flatten(input_shape=(256,256,1)),
    keras.layers.Dense(inputNeurons, activation=tf.nn.relu),
    #keras.layers.Dense(inputNeurons, activation=tf.nn.relu),
    #keras.layers.Dense(inputNeurons, activation=tf.nn.relu),
    #keras.layers.Dense(inputNeurons, activation=tf.nn.relu),
    keras.layers.Dense(7, activation = tf.nn.softmax)

    ])

    model.compile(
    
    optimizer = tf.optimizers.Adamax(learning_rate=0.001),
    loss  = "sparse_categorical_crossentropy", 
    metrics  = ["accuracy"]
    
    )



    model.fit(trainImages,trainLabels, 
          epochs=numepochs, 
          batch_size = 32)


    model.evaluate(testImages, testLabels)

    predicitions = model.predict(testImages[:])
   
    
    # use all



    guess = []
    same = 0.00
    guess = np.argmax(predicitions, axis=1) 
    count = 0
    end = time.time()

    testAccu =0
    for y in guess:
       # print(y)
        if testLabels[count] == y:
            same+=1
        count +=1

    testAccu = (same / 10925)*100


    print("test accuracy: ", testAccu)
    #dirName = 'C:/Users/Home/Desktop/faceAI/faceAI/models/' + str(run)
    #os.mkdir(dirName)
    #model.save(dirName)

    #return(tottime, accu,)


#ArtAI(128,1)
"""
times = end - start
#x = np.arange(0, 5, times)
y = np.sin(x)
plt.plot(x, y)
"""
#plt.show()


