from __future__ import division, print_function, absolute_import
import cv2
import numpy as np
from PIL import Image
import glob
from skimage.io import imread
from sklearn.svm import SVC
import os
import re
import cv2
from PIL import Image
import numpy as np
from sklearn.externals import joblib
import cPickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

labels = []
labels1 = []
things = []
testing1X = []
testingLabels = []
def getlabels(dir):
	global things
	subdirs = [x[0] for x in os.walk(dir)]
	for subdir in subdirs:
		files = os.walk(subdir).next()[2]

		if (len(files) > 0):
			for file in files:
				directoryName = subdir+"/"+file
				labelName = re.sub('\/home/bruce/Documents/honours/imageProccesing/testCharacters/','',subdir)
				if (labelName not in things):
					things.append(labelName)
	return things
labels1.append(getlabels("/home/bruce/Documents/honours/imageProccesing/testCharacters/"))

def list_filesTest(dir,array):
	global labels
	global testing1X
	global testingLabels
	subdirs = [x[0] for x in os.walk(dir)]
	for subdir in subdirs:
		files = os.walk(subdir).next()[2]

		if (len(files) > 0):
			for file in files:
				directoryName = subdir+"/"+file
				labelName = re.sub('\/home/bruce/Documents/honours/imageProccesing/testCharacters/','',subdir)

				testX = imread(directoryName,as_grey=True)
				testX = cv2.adaptiveThreshold(testX,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
							cv2.THRESH_BINARY,11,2)
				testing1X.append(testX.flatten())

				array.append(testX.flatten())
				labels.append(labelName)
				testingLabels.append(labelName)



with open('classifier.pkl', 'rb') as fid:
    clf = cPickle.load(fid)

for i in things:
	C=[]
	labels=[]
	list_filesTest("/home/bruce/Documents/honours/imageProccesing/testCharacters/"+i,C)
	testX=np.array(C,dtype="float32")
	testX=testX/255
	#print(clf.predict(testX))
	#print(labels)
	print(clf.score(testX,labels),labels,clf.predict(testX))

testingX = np.array(testing1X,dtype = 'float32')
testingX = testingX/255
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

y_pred= clf.predict(testingX)
labels = {'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'}
cnf_matrix = confusion_matrix(testingLabels, y_pred)
np.set_printoptions(precision=1)


plt.figure(figsize=(50,50))
plot_confusion_matrix(cnf_matrix, classes=labels, normalize = False,
                      title='Confusion matrix')

plt.show()
