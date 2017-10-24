
# import the necessary packages
import argparse
import cv2
import cPickle
import numpy as np
from imutils import contours
import imutils
import sys
refPt = []
cropping = False

import matplotlib.pyplot as plt
global contours

image = sys.argv[1]

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
answer = []
cords=[]
cords1 = []
groupLocs = []
cropping = False
d = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'0':0,'A':11,'B':13,'C':14,'D':15,'E':16,'F':17,'G':18,'H':19,'I':20,'J':21,'K':22,'L':23,'M':24,'N':25,'O':26,'P':27,'Q':28,'R':29,'S':30,'T':31,'U':32,'V':33,'W':34,'X':35,'Y':36,'Z':37}
blank_image = 0
cnts=[]
def guess(chars,cords,cords1):
    global answer
    with open('my_dumped_classifier.pkl', 'rb') as fid:
        clf = cPickle.load(fid)
    a = np.diff(cords)

    for i in range(len(cnts)):
        testArray = np.array(cnts[i],dtype="float32")
        testArray = testArray/255
        testArray=testArray.reshape(-1,784)


        predictions = clf.predict(testArray)
        for key in d:

            for x in np.nditer(predictions):
                if(d[key]==x.astype(int)):

					cv2.putText(blank_image,key, (cords[i],cords1[i]), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
					answer.append(key)


    #cv2.drawContours(img,contours,contourIndex,(0,255,0),3)

def findRegions(img):
	global groupLocs

	ref = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV |
		cv2.THRESH_OTSU)[1]

	refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
	refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]


	gradX = cv2.Sobel(ref, ddepth=cv2.CV_32F, dx=1, dy=0,ksize=-1)
	gradX = np.absolute(gradX)
	(minVal, maxVal) = (np.min(gradX), np.max(gradX))
	gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
	gradX = gradX.astype("uint8")

	rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
	output = []
	gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
	thresh = cv2.threshold(gradX, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	# apply a closing operation using the rectangular kernel to help
	# cloes gaps in between rounting and account digits, then apply
	# Otsu's thresholding method to binarize the image


	# find contours in the thresholded image, then initialize the
	# list of group locations
	groupCnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	groupCnts = groupCnts[0] if imutils.is_cv2() else groupCnts[1]


	# loop over the group contours
	for (i, c) in enumerate(groupCnts):
		# compute the bounding box of the contour
		(x, y, w, h) = cv2.boundingRect(c)

		# only accept the contour region as a grouping of characters if
		# the ROI is sufficiently large
		if w > 5 and h > 15 and h<100 :
			crop = img[y:y+h,x:x+w]
			groupLocs.append(crop)
			cv2.rectangle(blank_image,(x,y),(x+w,y+h),(255,255,255),1)

def findCharacters(img):
    global cords,blank_image,cnts
    images=[]

    thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    cv2.THRESH_BINARY,11,8)

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count =0
    blank_image = np.zeros((height,width,3), np.uint8)

    #sortContours(contours)


    #print(len(contours))
    for i in range(len(contours)):

        rect = cv2.boundingRect(contours[i])

        x,y,w,h = rect


        if(w*h>500 and w*h<1000):

        #cords.sort()
            crop = img[y:y+h,x:x+w]
            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
            crop = img[y:y+h,x:x+w]
            #crop = cv2.resize(crop, (28,28))
            cv2.imwrite("training-images/image-"+str(count)+".jpg",crop)
            count+=1

            crop = cv2.resize(crop,(28,28))
            crop = cv2.adaptiveThreshold(crop,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,11,0)
            cnts.append(crop.flatten())
            cords.append(x)
            cords1.append(y)
        #images.append(crop.flatten())
            #guess(crop.flatten(),x,y)
        #cv2.imwrite("characters/1new-image-"+str(count)+".jpg",crop)
        #cv2.putText(img,str(a), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)




        count+=1




    guess(images,cords,cords1)
    #cv2.imwrite("Characters.jpg",img)
#cropImageOnReceipt(image)


# load the image, clone it, and setup the mouse callback function


image = cv2.imread(image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image1 = image.copy()
height, width = image.shape[:2]
if(height>1000 and width >1000):
    width=int(round(width*0.3))
    height=int(round(height*0.3))

image = cv2.resize(image, (width,height))
cv2.imshow("Original",image)
cv2.waitKey(0)
findCharacters(image)
cv2.imshow("Characters",image)
cv2.waitKey(0)
findRegions(image1)

cv2.imshow("Newly Printed",blank_image)
cv2.waitKey(0)

#cv2.imshow("Text on image",image)
#cv2.waitKey(0)
# close all open windows
cv2.destroyAllWindows()
