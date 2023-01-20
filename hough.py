import cv2
import numpy as np
from PIL import Image

image='/Users/khelifibilel/Desktop/data/embryo_dataset/AG782-8/D2013.07.16_S0883_I132_WELL8_RUN100.jpeg'
img = cv2.imread(image, cv2.IMREAD_COLOR)
  
# Convert to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
# Blur using 3 * 3 kernel.
gray_blurred = cv2.blur(gray, (3, 3))
  
# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(gray_blurred,
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
               param2 = 30, minRadius = 100, maxRadius = 110)
a=0
b=0
r=0
# Draw circles that are detected.
if detected_circles is not None:
  
    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))
  
    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
        print(a,b,r)
        # Draw the circumference of the circle.
        cv2.circle(img, (a, b), r, (0, 255, 0), 2)
        
        im = Image.open(image)

        left = a-r-20
        top = b-r-20
        right =a+r+20
        bottom =b+r+20
        im1 = im.crop((left, top, right, bottom))
        im1.save('/Users/khelifibilel/Desktop/astronaut_pillow_crop.jpg', quality=95)

        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
        cv2.imshow("Detected Circle", img)
        cv2.waitKey(0)
        


