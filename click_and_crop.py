
# import the necessary packages
import argparse
import cv2
import cPickle
import numpy as np
from imutils import contours
import imutils
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
answer = []
cords=[]
cords1 = []
cropping = False
images=[]

def findCharacters(img):
	global x,cords,images
	count=0

	thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
	cv2.THRESH_BINARY,11,8)
	cv2.imshow("This",thresh)
	cv2.waitKey(0)
	contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


	img1 = img.copy()
	img2 = img.copy()


    #sortContours(contours)

	cnts=[]
	for i in range(len(contours)):
		rect = cv2.boundingRect(contours[i])
		x,y,w,h = rect
		if(w*h>500 and w*h<1600):
			cnts.append(contours[i])
			cords1.append(rect)
			cords.append(x)
        #cords.sort()


        #print(cnts)
	for a in range(len(cnts)):
		x,y,w,h = cords1[a]
		print(x,y)
		crop = img[y:y+h,x:x+w]

		crop = cv2.resize(crop, (28,28))
		crop = cv2.adaptiveThreshold(crop,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
				cv2.THRESH_BINARY,11,4)

        #images.append(crop.flatten())

		#cv2.imwrite("testCharacters/11235133new-image-"+str(count)+".jpg",crop)
		#cv2.putText(img,str(a), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2)
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
		images.append(crop.flatten())
		count+=1

    #guess(images)
    #cv2.imwrite("Characters.jpg",img)
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables

	global refPt, cropping
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False

		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", roi)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"],0)
height, width = image.shape[:2]
if(height>1000 and width >1000):
    width=int(round(width*0.3))
    height=int(round(height*0.3))

image = cv2.resize(image, (width,height))
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF

	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()

	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break

# if there are two reference points, then crop the region of interest
# from teh image and display it
if len(refPt) == 2:
	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	#cv2.imshow("ROI", roi)
	cv2.waitKey(0)
findCharacters(roi)

image = cv2.imread(args["image"])
height, width = image.shape[:2]
if(height>1000 and width >1000):
    width=int(round(width*0.3))
    height=int(round(height*0.3))
image = cv2.resize(image, (width,height))
cv2.imwrite("this.jpg",roi)

# close all open windows
cv2.destroyAllWindows()
