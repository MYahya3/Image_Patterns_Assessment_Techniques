import cv2
import numpy as np
from skimage.metrics import structural_similarity

img1 = cv2.imread('img1.png')
img2 = cv2.imread("img3.png")

img1 = cv2.resize(img1, (480,480))
img2 = cv2.resize(img2, (480,480))

ROI = cv2.selectROI("select the area",img1)
x, y, w, h = ROI

img1 = img1[int(y):int(y+w), int(x):int(x+h)]
img2 = img2[int(y):int(y+w), int(x):int(x+h)]

result1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
result2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

(score, diff) = structural_similarity(result1, result2, full=True)
print("Image similarity", score)

diff = (diff * 255).astype("uint8")

# Threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

mask = np.zeros(img1.shape, dtype='uint8')
filled_after = img2.copy()

for c in contours:
    area = cv2.contourArea(c)
    print(area)
    if area > 40:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0,0,255), 1)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0,0,255), 1)
        cv2.drawContours(mask, [c], 0, (0,255,0), -1)
        cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)


cv2.imshow('result', np.hstack([img1,img2]))
cv2.imshow('diff',diff)
cv2.imshow('mask',mask)
cv2.imshow('filled after',filled_after)
# cv2.imshow('result', res2)

# cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
