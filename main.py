
# import the necessary packages
import argparse
import cv2
import cPickle
import numpy as np
from imutils import contours
import imutils
import sys
import math as Math
import pdb
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
import subprocess
import sys
import difflib
global contours
refPt = []
cropping = False
image2 = sys.argv[1]

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
answer = []
cords=[]
cords1=[]
groupLocs = []
chars1 = []
wordCharacters = []
cropping = False
d = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'A':10,'B':11,'C':12,'D':13,'E':14,'F':15,'G':16,'H':17,'I':18,'J':19,'K':20,'L':21,'M':22,'N':23,'O':24,'P':25,'Q':26,'R':27,'S':28,'T':29,'U':30,'V':31,'W':32,'X':33,'Y':34,'Z':35}
blank_image = 0
cnts=[]
wordImages = []
images=[]
words = []

'''
Adapted from a tutorial on PyImageSearch. https://www.pyimagesearch.com/2015/11/30/detecting-machine-readable-zones-in-passport-images/
'''

def findRegions(img):
	global groupLocs,wordImages


	thresh1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
				cv2.THRESH_BINARY,11,17)

	ref = cv2.threshold(thresh1, 0, 255, cv2.THRESH_BINARY_INV |
		cv2.THRESH_OTSU)[1]
	cv2.imshow("Thresh",ref)
	cv2.waitKey(0)
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


	# loop over the group contours
	for c in (groupCnts):
		# compute the bounding box of the contour
		(x, y, w, h) = cv2.boundingRect(c)


		if w > 5 and h > 15 and h<100 :
			crop = img[y-20:y+h+20,x-20:x+w+20]

			cv2.rectangle(imageCopy,(x,y),(x+w,y+h),(0,0,255),3)
			np.unpackbits(crop)

			if(crop.size):
				wordImages.append(crop)
	cv2.imshow("Regions",imageCopy)
	cv2.waitKey(0)
	cv2.imwrite("Thing.jpg",imageCopy)

def findCharacters(img):
	global cords
	cords1 = []

	#finding
	thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
				cv2.THRESH_BINARY,11,8)

	contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	count =0

	for i in range(len(contours)):
		rect = cv2.boundingRect(contours[i])

		x,y,w,h = rect


		if(w*h>450	 and w*h<1300):
			crop = img[y:y+h,x:x+w]
			#cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
            #crop = cv2.resize(crop, (28,28))
            #cv2.imwrite("training-images/image-"+str(count)+".jpg",crop)
			count+=1
			crop = cv2.resize(crop,(28,28))
			crop = cv2.adaptiveThreshold(crop,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
					cv2.THRESH_BINARY,11,0)

			cords1.append(x)


			images.append(crop.flatten())
			#cv2.imwrite("characters/1new-image-"+str(count)+".jpg",crop)
			#cv2.putText(img,str(a), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2)

	if(len(cords1)>0):
		cords.append(cords1)

	return images




def guess(chars):

	with open('classifier.pkl', 'rb') as fid:
		clf = cPickle.load(fid)
	answer=[]
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

def correctWords(a,b):
    return difflib.SequenceMatcher(None, b, a).ratio()

image = cv2.imread(image2)


height, width = image.shape[:2]
if(height>1000 and width >1000):
    width=int(round(width*0.3))
    height=int(round(height*0.3))

image = cv2.resize(image, (width,height))
imageCopy = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original",image)
cv2.waitKey(0)
cv2.imwrite("copy.jpg",image)



#guess(images,cords,cords1)

findRegions(image)

#findCharacters(wordImages[0])
for i in range (0,len(wordImages)-1):
	images=[]

	answer=[]
	if(len(wordImages[i]>0)):
		#print(wordImages[i])

		wordCharacters.append(findCharacters(wordImages[i]))
	#print(wordCharacters)
	#print(answer)
	#print(wordImages[i])

for i in range(len(wordCharacters)):
	if(len(wordCharacters[i])>0):
		words.append(guess(wordCharacters[i]))

itemsBought = []
itemsBought.append("copy.jpg")
for i in range(len(words)):

	grouped = zip(cords[i],words[i])
	sorting=sorted(grouped)
	finalSorting = [point[1] for point in sorting]
	print(''.join(finalSorting))
	with open("items.txt", "r") as ifile:
		for line in ifile:

			if((correctWords(''.join(finalSorting),line))>=0.35):
				if line not in itemsBought:
					itemsBought.append(line)
total =0

for i in itemsBought[1:]:
	[int(s) for y in i.split(' ') if y.isdigit() ]
	total +=float(y)

itemsBought.append("Total: R{:0.2f}\n".format(total))
f=open('itemsBought.txt','w')
for items in itemsBought:
    f.write(items+'\n')
f.close()


subprocess.Popen(["python", "gui2.py"])
sys.exit(0)
	#print(''.join(finalSorting))
#for x in wordCharacters:
#	print(guess(x))
#for j in range(len(wordCharacters)):
	#cv2.imshow("image",wordCharacters[j].flatten())
#	cv2.waitKey(0)
	#print(guess(wordCharacters[j]))
cv2.imwrite("regions.jpg",image)

#cv2.imshow("Newly Printed",blank_image)
#cv2.waitKey(0)
#cv2.imwrite("image.jpg",blank_image)
cv2.destroyAllWindows()
