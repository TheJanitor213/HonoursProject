
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
d = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'0':0,'A':10,'B':11,'C':12,'D':13,'E':14,'F':15,'G':16,'H':17,'I':18,'J':19,'K':20,'L':21,'M':22,'N':23,'O':24,'P':25,'Q':26,'R':27,'S':28,'T':29,'U':30,'V':31,'W':32,'X':33,'Y':34,'Z':35}
blank_image = 0
cnts=[]
def guess(chars,cords,cords1):
    #global answer
    with open('my_dumped_classifier.pkl', 'rb') as fid:
        clf = cPickle.load(fid)

    for i in range(len(cnts)):
        testArray = np.array(cnts[i],dtype="float32")
        testArray = testArray/255
        testArray=testArray.reshape(-1,784)


        predictions = clf.predict(testArray)
        for key in d:

            for x in np.nditer(predictions):
                if(d[key]==x.astype(int)):


					#cv2.putText(blank_image,key, (cords[i],cords1[i]), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

					cv2.imwrite("characters/"+str(d[key]-2)+"/img4234-"+str(i)+".jpg",cnts[i].reshape(28,28))
					#answer.append(key)
					#print(key)

def findCharacters(img):

    global cords,blank_image,cnts
    images=[]

    thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    cv2.THRESH_BINARY,11,10)

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count =0
    blank_image = np.zeros((height,width,3), np.uint8)

    #sortContours(contours)


    #print(len(contours))
    for i in range(len(contours)):

        rect = cv2.boundingRect(contours[i])

        x,y,w,h = rect

        if(w*h>400 and w*h<1000):

        #cords.sort()
            crop = img[y:y+h,x:x+w]

            crop = cv2.resize(crop,(28,28))
            crop = cv2.adaptiveThreshold(crop,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,11,5)
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
cv2.imshow("Original",image)
cv2.waitKey(0)
findCharacters(image)
cv2.imshow("Characters",image)
cv2.waitKey(0)

#cv2.putText(image,''.join(finalAnswer1), refPt[1], cv2.FONT_HERSHEY_SIMPLEX, 5,(180,0,100),2)

#cv2.imshow("Text on image",image)
#cv2.waitKey(0)
# close all open windows
cv2.destroyAllWindows()
