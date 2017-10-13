import os
import re
import cv2
from PIL import Image
import numpy as np
svmTrain = []

def data(source,label):
    divisor=255
    img = cv2.imread(source,0)
    width, height = img.shape
    thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    string =  str(label)+ " "

    featureCount = 1
    for i in range(height):
        for j in range(width):
            string += str(featureCount)+":"+str(img[i,j]/divisor)+" "

            featureCount+=1

    svmTrain.append(string)
    return string
def list_files(dir):
    r = []
    subdirs = [x[0] for x in os.walk(dir)]
    count = 1

    for subdir in subdirs:
        files = os.walk(subdir).next()[2]

        if (len(files) > 0):
            for file in files:
                directoryName = subdir+"/"+file
                labelName = re.sub('\/home/bruce/Documents/honours/imageProccesing/characters/','',subdir)
                data(directoryName,labelName)


list_files("/home/bruce/Documents/honours/imageProccesing/characters")
print(svmTrain)
thefile = open('dataTrain.txt', 'w')
for i in svmTrain:
     thefile.write("%s\n" % i)
