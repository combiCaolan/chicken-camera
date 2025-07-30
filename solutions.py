import cv2
import numpy as np

# Load your image
img = cv2.imread('test-data/gimagesphoto.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

cv2.imshow('Original', img)
cv2.imshow('Grayscale', gray)
cv2.imshow('Threshold', thresh)

# Wait for key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()