
import os
import random
import shutil


dir = 'C:/Users/Home/Desktop/FERG_DB_256/ray/ray_surprise'
newdir = 'C:/Users/Home/Desktop/faceAI/faceAI/Cohn-Kanade Images/Train/surprise'
testdir = 'C:/Users/Home/Desktop/faceAI/faceAI/Cohn-Kanade Images/Test/surprise'

files = os.listdir(dir)

size=  len(files)


psize = (size/ 10) *8

while psize >0:
   
    filename = str(dir + "/"+ random.choice(os.listdir(dir)))
    shutil.move(filename, newdir)
    psize -=1

files = os.listdir(dir)
for x in range(len(files)):
    filename = str(dir + "/"+ random.choice(os.listdir(dir)))
    shutil.move(filename, testdir)


   
