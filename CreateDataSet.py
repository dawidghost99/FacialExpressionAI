"""
import os
import csv
entries = os.listdir('img/')



with open('results.csv', 'w') as file:
   writer = csv.writer(file)
   for x in entries:
        
       s = ''.join([i for i in x if not i.isdigit()])
       writer.writerow([s,x])
"""


"""
Dataset provided by - https://grail.cs.washington.edu/projects/deepexpr/ferg-db.html
This contains 55767 annotated face images of six stylized characters modeled using MAYA software
"""

import numpy as np
import random
import cv2
import pandas as pd
import os
import pickle
from tqdm import tqdm


#laptop: C:/Users/Home/Desktop/faceAI/faceAI/img
#PC: C:/Users/dawid/Desktop/FaceAI/FaceAI/FaceAI/img


def expressions():

    CATEGORIES = ["anger","disgust","fear","joy","neutral","sadness","surprise"]


    training_data = []
    testing_data = []
    IMG_SIZE = 256


    def create_training_data():


         Dataloc  = "C:/Users/Home/Desktop/faceAI/faceAI/Cohn-Kanade Images/TRAIN"

         for category in CATEGORIES:  
            path = os.path.join(Dataloc,category)  # create path 
            for img in os.listdir(path):  # iterate over each image per emotion
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array



         

         new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))



        
         for category in CATEGORIES:

            path = os.path.join(Dataloc,category)  # create path
            class_num = CATEGORIES.index(category)  # get the classification 

            for img in tqdm(os.listdir(path)):  # iterate over each image per emoition
                try:
                    img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                    training_data.append([new_array, class_num])  # add this to the training_data
                except Exception as e:  # in the interest in keeping the output clean...
                    pass



    def create_testing_data():



         Testloc = "C:/Users/Home/Desktop/faceAI/faceAI/Cohn-Kanade Images/TEST"

         for category in CATEGORIES:  
            path = os.path.join(Testloc,category)  # create path 
            for img in os.listdir(path):  # iterate over each image per emotion
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
               



         

         new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))


         for category in CATEGORIES: 

            path = os.path.join(Testloc,category)  # create path 
            class_num = CATEGORIES.index(category)  # get the classification  

            for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
                try:
                    img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                    testing_data.append([new_array, class_num])  # add this to our training_data
                except Exception as e:  # in the interest in keeping the output clean...
                    pass

    
    create_training_data()
    create_testing_data()



    random.shuffle(training_data)
    random.shuffle(testing_data)


    train_images= [] #images
    train_lables = [] #lables

    testing_images = []
    testing_lables = []

    for features,label in training_data:
       train_images.append(features)
       train_lables.append(label)



    for features,label in testing_data:
       testing_images.append(features)
       testing_lables.append(label)

   # print(train_images[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

    train_images= np.array(train_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    testing_images= np.array(testing_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
   


      #pickel for testing set 
    

    pickle_out = open("testing_images.pickle","wb")
    pickle.dump(testing_images, pickle_out)
    pickle_out.close()

    pickle_out = open("testing_lables.pickle","wb")
    pickle.dump(testing_lables, pickle_out)
    pickle_out.close()

    pickle_in = open("testing_images.pickle","rb")
    testing_images= pickle.load(pickle_in)

    pickle_in = open("testing_lables.pickle","rb")
    testing_lables = pickle.load(pickle_in)

    #pickel for training set

    pickle_out = open("train_images.pickle","wb")
    pickle.dump(train_images, pickle_out)
    pickle_out.close()

    pickle_out = open("train_lables.pickle","wb")
    pickle.dump(train_lables, pickle_out)
    pickle_out.close()

    pickle_in = open("train_images.pickle","rb")
    train_images= pickle.load(pickle_in)

    pickle_in = open("train_lables.pickle","rb")
    train_lables = pickle.load(pickle_in)


