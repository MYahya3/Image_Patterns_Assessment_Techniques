import cv2
import numpy as np
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error as compare_mse

image = cv2.imread('ref.png')
image = cv2.resize(image, (480,480))
ROI = cv2.selectROI("select the area",image, False)
x, y, w, h = ROI
ref = image[int(y):int(y+w), int(x):int(x+h)]
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)


path = "video.mp4"
# Initialize the webcam feed
capture = cv2.VideoCapture(path)
count = 0
# Main Logic
while True:

    # Start reading the webcam feed frame by frame
    ret, frame = capture.read()
    if not ret:
        break
    frame = cv2.resize(frame, (480,480))
    frame_roi = frame[int(y):int(y+w), int(x):int(x+h)]
    frame_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    (score, diff) = structural_similarity(frame_roi, ref, full=True)
    mse = compare_mse(frame_roi, ref)
    print(f"SS {score}, MSE {mse}")
    diff = (diff * 255).astype("uint8")
    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # print(thresh)
    count += 1
    # if (score)*100 < 50:
    #     print(f"Data DRIFT {int(score*100)}%")
    # else:
    #     pass
    cv2.imshow("images", np.hstack([frame, image]))
    key = cv2.waitKey(0)
    if key == 27:
        break
print(count)
cv2.destroyAllWindows()