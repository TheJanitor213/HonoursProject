from skimage.segmentation import clear_border
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# find contours in the MICR image (i.e,. the outlines of the
# characters) and sort them from left to right
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

# apply a closing operation using the rectangular kernel to help
# cloes gaps in between rounting and account digits, then apply
# Otsu's thresholding method to binarize the image


# find contours in the thresholded image, then initialize the
# list of group locations
groupCnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
groupCnts = groupCnts[0] if imutils.is_cv2() else groupCnts[1]
groupLocs = []

# loop over the group contours
for (i, c) in enumerate(groupCnts):
	# compute the bounding box of the contour
	(x, y, w, h) = cv2.boundingRect(c)

	# only accept the contour region as a grouping of characters if
	# the ROI is sufficiently large
	if w > 5 and h > 15 and h<100 :
		groupLocs.append((x, y, w, h))
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,0),1)

# sort the digit locations from left-to-right
groupLocs = sorted(groupLocs, key=lambda x:x[0])
cv2.imshow("groups",image)
cv2.waitKey(0)
cv2.imwrite("proper.jpg",image)
