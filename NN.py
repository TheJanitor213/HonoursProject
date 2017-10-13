# -*- coding: utf-8 -*-

""" Convolutional Neural Network for MNIST dataset classification task.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""

from __future__ import division, print_function, absolute_import
import cv2

import numpy as np
from PIL import Image
import glob
import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from skimage.io import imread
from sklearn.svm import SVC
# Data loading and preprocessing
import os
import re
import cv2
from PIL import Image
import numpy as np
from sklearn.externals import joblib
import cPickle

svmTrain = []

X,Y = [],[]
B=[]
C=[]
D=[]
E=[]
F=[]
G=[]
#X_train= np.zeros()
testX,testY=[],[]

def list_filesTrain(dir,array):
    global B
    global D
    global F
    subdirs = [x[0] for x in os.walk(dir)]

    count = 1

    for subdir in subdirs:
        files = os.walk(subdir).next()[2]

        if (len(files) > 0):
            for file in files:
                directoryName = subdir+"/"+file
                labelName = re.sub('\/home/bruce/Documents/honours/imageProccesing/characters/','',subdir)

                X= imread(directoryName,as_grey=True)
                X = cv2.adaptiveThreshold(X,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                        cv2.THRESH_BINARY,11,2)
                array.append(X.flatten())


                D.append(labelName)
        F.append(D)
        D=[]



list_filesTrain("/home/bruce/Documents/honours/imageProccesing/characters",B)

def list_filesTest(dir,array):
    global C
    global E
    global G
    subdirs = [x[0] for x in os.walk(dir)]
    count = 1
    for subdir in subdirs:
        files = os.walk(subdir).next()[2]

        if (len(files) > 0):
            for file in files:
                directoryName = subdir+"/"+file
                labelName = re.sub('\/home/bruce/Documents/honours/imageProccesing/testCharacters/','',subdir)

                testX = imread(directoryName,as_grey=True)
                testX = cv2.adaptiveThreshold(testX,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                        cv2.THRESH_BINARY,11,2)
                array.append(testX.flatten())
                E.append(labelName)


list_filesTest("/home/bruce/Documents/honours/imageProccesing/testCharacters",C)
'''
thefile = open('train.txt', 'w')
for i in svmTrain:
     thefile.write("%s\n" % i)
'''
F=F[1:]
G=G[1:]

tensorarray = []
X=np.array(B,dtype="float32")
testX=np.array(C,dtype="float32")


F=np.array(F)
F=F.flatten()

for i in F:

    for j in i:
        Y.append(j)

clf = SVC()
clf.fit(X,Y)

# save the classifier
