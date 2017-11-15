
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
d = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'0':0,'A':11,'B':13,'C':14,'D':15,'E':16,'F':17,'G':18,'H':19,'I':20,'J':21,'K':22,'L':23,'M':24,'N':25,'O':26,'P':27,'Q':28,'R':29,'S':30,'T':31,'U':32,'V':33,'W':34,'X':35,'Y':36,'Z':37}
words = []
def guess(imgs):
    global answer
    with open('my_dumped_classifier.pkl', 'rb') as fid:
        clf = cPickle.load(fid)

    testArray = np.array(imgs,dtype="float32")
    testArray = testArray/255

    predictions = clf.predict(testArray)
    for key in d:

        for x in np.nditer(predictions):
            if(d[key]==x.astype(int)):

				answer.append(key)
				cv2.imwrite("characters/img4234-"+str(i)+".jpg",imgs.reshape(28,28))


'''def sortContours(cnts):
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][0], reverse=False))
    print("working")
    # return the list of sorted contours and bounding boxes
    return (cnts)
'''

def findCharacters(img):
    global x,cords,images


    thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    cv2.THRESH_BINARY,11,8)

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    img1 = img.copy()
    img2 = img.copy()


    #sortContours(contours)

    cnts=[]
    for i in range(len(contours)):
        rect = cv2.boundingRect(contours[i])
        x,y,w,h = rect
        if(w*h>500 and w*h<1300):
            cnts.append(contours[i])
            cords1.append(rect)
            cords.append(x)
        #cords.sort()

    p = zip(cords,cnts)
    p=sorted(p)
    P = [point[1] for point in p]

        #print(cnts)
    for a in range(len(P)):
        x,y,w,h = cords1[a]
        print(x,y)
        crop = img[y:y+h,x:x+w]

        crop = cv2.resize(crop, (28,28))
        crop = cv2.adaptiveThreshold(crop,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,4)

        #images.append(crop.flatten())

        #cv2.imwrite("characters/1new-image-"+str(count)+".jpg",crop)
        cv2.putText(img,str(a), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
        images.append(crop.flatten())


    cv2.imshow("Characters",img)

    cv2.waitKey(0)
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
		cv2.imshow("image", image)


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
guess(images)
image = cv2.imread(args["image"])
height, width = image.shape[:2]
if(height>1000 and width >1000):
    width=int(round(width*0.3))
    height=int(round(height*0.3))
image = cv2.resize(image, (width,height))

cordsanswer, finalAnswer1 = zip(*sorted(zip(cords, answer)))
finalAnswer = []


cv2.putText(image,''.join(finalAnswer1), refPt[1], cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)

cv2.imshow("Text on image",image)
cv2.waitKey(0)
# close all open windows
cv2.destroyAllWindows()
