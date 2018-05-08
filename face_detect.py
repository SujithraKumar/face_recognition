import cv2
import numpy as np
import sys


# Get user supplied values
imagePath = 'D:/image recoginition over 1 km/important code/FaceDetect/detects/jpg/ima17.jpg'  
cascPath = 'D:/image recoginition over 1 km/important code/FaceDetect/haarcascade_frontalface_default.xml'

print imagePath
# Create the haar cascadeima28
faceCascade = cv2.CascadeClassifier(cascPath)

print "faceCascade value"
print faceCascade

# Read the image
image = cv2.imread(imagePath)

print "image value"
print image

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print "Gray value"
print gray


# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(5, 5),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

#faces = faceCascade.detectMultiScale(gray, 1.1, 5)

print "number of faces"
print len(faces)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
