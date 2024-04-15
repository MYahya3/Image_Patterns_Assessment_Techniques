import cv2
import numpy as np

def resize(img, dim=(720, 680)):
    img = cv2.resize(img, dim)
    return img

def compare_image(image, ref_image, method):
    result = cv2.matchTemplate(image, ref_image, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_val, max_loc

def ref_roi(image,roi_list=None, roi=False):
    # Load the reference image
    ref_image = cv2.imread(image)
    ref_image = resize(ref_image)
    # Convert the reference image to grayscale
    gray_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    # ROI Coordinates
    # Define the region of interest in the reference image
    if roi == True: # If there is define REGION OF INTEREST withing REF Image
        x, y, w, h = roi_list[0], roi_list[1], roi_list[2], roi_list[3]
        gray_img = gray_image[y:y + h, x:x + w]  # adjust the coordinates and size to your specific ROI
        cv2.rectangle(ref_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    else:
        gray_img = gray_image # Else compare complete REF Image
    return ref_image , gray_img

# file_name = os.path.basename(os.path.splitext(file_path)[0])
def data_drift(file_path, ref_image,Num_Frames, gray_img):
    # csv_path = "file.csv"
                        # Convert reference image into grayscale and set roi if Coordinated given #
    frames_counter = 0 # To count total frame on which we evaluate camera angel
    counterlist = [] # List to save 0 if no cam_angel drift else 1 e.g [1,1,1,0,0]

    cap = cv2.VideoCapture(file_path)

                                ### While Loop for video frames ###
    while True:
                                        # Read a frame from the video
        ret, frame = cap.read()
        fps_5 = int(cap.get(cv2.CAP_PROP_FPS))

        # If no frame returns loop break
        if not ret: break

                            # To confirm if there is frame then resize it else pass it
        if (type(frame) == type(None)): pass
        else:   frame = resize(frame)

                            # To get and compare only frame after 5 seconds in video
        frames_counter += 1
        if frames_counter % fps_5 == 0:
            frame = frame
                                    # Convert the video frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                            # Use template matching to locate the ROI in the video frame
            max_val, max_loc = compare_image(gray_frame, ref_image=gray_img, method=cv2.TM_CCOEFF_NORMED)

                # To Check if the ROI is in the same position in the last video frame as in the reference image

            # Draw a rectangle around the ROI in the video frame
            top_left = max_loc
            bottom_right = (top_left[0] + gray_img.shape[1], top_left[1] + gray_img.shape[0])
            if max_val < 0:
                max_val = 0
            print(f"MAX VALUE : {max_val}")

                        # Set Simlarity thresholdbetween Reference ROI and Video Frame to check if any drift #
            if (max_val) * 100 < 90:
                counterlist.append(1)
            else:
                counterlist.append(0)
                # cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
                # cv2.putText(frame, 'CAM ANGEL OKEY', (0, 40), font, fontScale, (0, 255, 0), thickness)
        else:   pass

                             # If data drift found =< 5 frame it will display TEXT OF DATA DRIFT FOUND #
        drift_counter = 0
        for num in counterlist:
            drift_counter += num
        # print(counterlist)
        # print(f"drift_counter:{drift_counter}")
                                    # Start Displaying Status after 4 frames evaluated #
        if len(counterlist) > Num_Frames - 1:
            if drift_counter >= Num_Frames:
                cv2.rectangle(frame, (2, 0), (160, 68), (255, 255, 255), -1)
                cv2.putText(frame, 'CAM ANGLE DRIFT', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,cv2.LINE_AA)
                cv2.putText(frame, f'Similarity : {int(max_val*100)}%', (14, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (2, 0), (160, 68), (255, 255, 255), -1)
                cv2.putText(frame, 'CAM ANGLE OKEY', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,cv2.LINE_AA)
                cv2.putText(frame, f'Similarity : {int(max_val*100)}%', (14, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,cv2.LINE_AA)
            if len(counterlist) > Num_Frames:
                counterlist.pop(0)
            else:
                pass
            # print(counterlist)
        # cv2.rectangle(ref_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Display the video frame with the ROI location marked
        cv2.imshow('Video Frame', np.hstack([ref_image, frame]))
        if cv2.waitKey(100) & 0xFF == ord('q'):
            # Press 'q' to quit
            break
    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()

# ref_path = "D:/skin_color_detection_opencv/ref.png"
# file_path = "D:/skin_color_detection_opencv/152809.mp4"

roi_list=[193, 56, 22, 22] # Defined in Reference Image

def main():
    ref_image, gray_img = ref_roi(image=ref_path, roi_list=roi_list, roi=True) # Set roi True if There is roi in ref image
    data_drift(file_path,ref_image, Num_Frames=7, gray_img=gray_img) # Num_Frames is threshold if drift comes in 7 continuous frames then Visualize it DRIFT

main()