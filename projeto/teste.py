import numpy as np
import cv2

# read original image
image = cv2.imread("dataset/4.png")

# convert to gray scale image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# contrast stretching 
# Function to map each intensity level to output intensity level. 
pixelVal = lambda pix, r1, s1, r2, s2: ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1

    # Define parameters. 


r1 = 70
s1 = 0
r2 = 100
s2 = 255

# Vectorize the function to apply it to each value in the Numpy array. 
pixelVal_vec = np.vectorize(pixelVal)

# Apply contrast stretching. 
contrast_stretched = pixelVal_vec(gray, r1, s1, r2, s2)


img_binary = np.where(contrast_stretched >= 125, 0, 255).astype(np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
img_dilated = cv2.dilate(img_binary, kernel, iterations=15)
img_eroded = cv2.erode(img_dilated, kernel, iterations=15)

cv2.imwrite('contrast_stretch.png', img_eroded)
