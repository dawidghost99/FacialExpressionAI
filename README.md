# FacialExpressionAI


8020split.py will split the data into 2 folders. It will put 80% of the data into a training folder and 20% into a testing folder. If
the number of data is not a even number (e.g. 4001 images) it will add the extra image to the training folder.

CreateDataset.py will prepare and create the dataset for the Neural network to train and test on. I will change the image sizes to 256x256 pixels
if they weren't that originaly. Then it grayscales the images. It outputs the Testing data, Testing lables, Traing data and training labels
in a pickle format so that the dataset doesn't need to be created everytime the neural network is used.

NN.py is the Artifical neural network.

FaceAI.py is the main python file that call the Neural network and gives it the needed parameters.
