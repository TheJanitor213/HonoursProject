import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
global contours

img = cv2.imread('receipts/picknpay.jpg')
image_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
image_y = np.zeros(image_yuv.shape[0:2],np.uint8)
image_y[:,:] = image_yuv[:,:,0]
cv2.imshow("Image",image_yuv)
cv2.waitKey(0)
image_blurred = cv2.GaussianBlur(image_y,(3,3),0)
edges = cv2.Canny(image_blurred,100,300,apertureSize=3)
cv2.imshow("Image",edges)
cv2.waitKey(0)
contours,hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    hull = cv2.convexHull(cnt)
simplified_cnt = cv2.approxPolyDP(hull,0.001*cv2.arcLength(hull,True),True)

for i in range(len(contours)):
    rect = cv2.boundingRect(contours[i])
    x,y,w,h = rect
    cv2.rectangle(edges,(x,y),(x+w,y+h),(0,255,0),1)
cv2.imshow("Characters ",edges)
cv2.waitKey(0)

#(H,mask) = cv2.findHomography(cnt.astype('single'),np.array([[[0., 0.]],[[2150., 0.]],[[2150., 2800.]],[[0.,2800.]]],dtype=np.single))

#final_image = cv2.warpPerspective(image,H,(2150, 2800))
#image_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
