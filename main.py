
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
import pdb
import matplotlib.pyplot as plt
global contours

image2 = sys.argv[1]

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
answer = []
cords=[]
cords1 = []
groupLocs = []
chars1 = []
wordCharacters = []
cropping = False
d = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'0':0,'A':11,'B':13,'C':14,'D':15,'E':16,'F':17,'G':18,'H':19,'I':20,'J':21,'K':22,'L':23,'M':24,'N':25,'O':26,'P':27,'Q':28,'R':29,'S':30,'T':31,'U':32,'V':33,'W':34,'X':35,'Y':36,'Z':37}
blank_image = 0
cnts=[]
wordImages = []

def findRegions(img):
	global groupLocs,wordImages
	image = cv2.imread("testing.jpg")
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	height, width = image.shape[:2]
	if(height>1000 and width >1000):
	    width=int(round(width*0.3))
	    height=int(round(height*0.3))

	image = cv2.resize(image, (width,height))

	ref = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV |
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


	groupCnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	groupCnts = groupCnts[0] if imutils.is_cv2() else groupCnts[1]

	groupLocs = []

	# loop over the group contours
	for c in (groupCnts):
		# compute the bounding box of the contour
		(x, y, w, h) = cv2.boundingRect(c)


		if w > 5 and h > 15 and h<100 :
			crop = img[y-20:y+h+20,x-20:x+w+20]
			#groupLocs.append((x, y, w, h))
			#cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),1)
			np.unpackbits(crop)
			if(crop!=""):
				wordImages.append(crop)
	#cv2.imshow("this",img)
	#cv2.waitKey(0)

def findCharacters(img):
	global cords,cords1
	images =[]

	thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
				cv2.THRESH_BINARY,11,8)

	contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	count =0

	for i in range(len(contours)):
		rect = cv2.boundingRect(contours[i])

		x,y,w,h = rect


		if(w*h>450	 and w*h<1000):
			crop = img[y:y+h,x:x+w]
            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
            #crop = cv2.resize(crop, (28,28))
            #cv2.imwrite("training-images/image-"+str(count)+".jpg",crop)
			count+=1
			crop = cv2.resize(crop,(28,28))
			crop = cv2.adaptiveThreshold(crop,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
					cv2.THRESH_BINARY,11,0)

			cords.append(x)
			cords1.append(y)

			images.append(crop.flatten())
			#cv2.imwrite("characters/1new-image-"+str(count)+".jpg",crop)
			#cv2.putText(img,str(a), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2)
			#cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
			count+=1
#print(len(images))
	guess(images)
	#print("this")


def guess(chars):

	print("tis")

	with open('my_dumped_classifier.pkl', 'rb') as fid:
		clf = cPickle.load(fid)

	for i in range(len(chars)):

		testArray = np.array(chars[i],dtype="float32")

		testArray = testArray/255

		testArray=testArray.reshape(-1,784)


        predictions = clf.predict(testArray)
        for key in d:

            for x in np.nditer(predictions):
                if(d[key]==x.astype(int)):
					#cv2.putText(blank_image,key, (cords[i],cords1[i]), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
					answer.append(key)

	return answer


image = cv2.imread(image2)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

height, width = image.shape[:2]
if(height>1000 and width >1000):
    width=int(round(width*0.3))
    height=int(round(height*0.3))

image = cv2.resize(image, (width,height))
#cv2.imshow("Original",image)
#cv2.waitKey(0)


#guess(images,cords,cords1)

findRegions(image)

# 	findCharacters(wordImages[0])
for i in range (1,len(wordImages)-1):
	answer=[]
	findCharacters(wordImages[i])
	print(answer)
	#print(wordImages[i])
	#print(wordCharacters)


#for x in wordCharacters:
#	print(guess(x))
#for j in range(len(wordCharacters)):
	#cv2.imshow("image",wordCharacters[j].flatten())
#	cv2.waitKey(0)
	#print(guess(wordCharacters[j]))


#cv2.imshow("Newly Printed",blank_image)
#cv2.waitKey(0)
#cv2.imwrite("image.jpg",blank_image)
cv2.destroyAllWindows()
