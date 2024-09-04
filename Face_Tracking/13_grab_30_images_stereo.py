import cv2
import os

# Set the directory where images will be saved
save_directory_right = 'calibration_images/stereo_right'
os.makedirs(save_directory_right, exist_ok=True)

save_directory_left = 'calibration_images/stereo_left'
os.makedirs(save_directory_left, exist_ok=True)

# Initialize the camera (default camera is usually index 0)
#camera = cv2.VideoCapture(0)
camera_right = cv2.VideoCapture(0) # right camera
camera_left = cv2.VideoCapture(1) #left camera

# Check if the camera opened successfully
if not camera_right.isOpened():
    print("Error: Could not open camera right.")
    exit()

if not camera_left.isOpened():
    print("Error: Could not open camera right.")
    exit()

image_count = 0
image_count_max = 30
checkerboard_size = (10,7)



while True:
    # Capture frame-by-frame
    ret_right, frame_right_original = camera_right.read()
    ret_left, frame_left_original = camera_left.read()

    #duplicate image
    frame_right = frame_right_original.copy()
    frame_left = frame_left_original.copy()

    if not ret_right:
        print("Error: Failed to capture image right.")
        break

    if not ret_left:
        print("Error: Failed to capture image left.")
        break

    # Convert the frame to grayscale
    gray_frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    gray_frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)

    # Find the checkerboard corners in the frame
    found_right, corners_right = cv2.findChessboardCorners(gray_frame_right, checkerboard_size, None)
    found_left, corners_left = cv2.findChessboardCorners(gray_frame_left, checkerboard_size, None)

    # If a checkerboard is found, draw it on the frame - Right Frame
    if found_right:
        cv2.drawChessboardCorners(frame_right, checkerboard_size, corners_right, found_right)

    # If a checkerboard is found, draw it on the frame - Right Frame
    if found_left:
        cv2.drawChessboardCorners(frame_left, checkerboard_size, corners_left, found_left)

    # write information on screen
    detection_status_right = "Checkerboard Detected" if found_right else "No Checkerboard"
    cv2.putText(frame_right, detection_status_right, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame_right, f'Images Captured: {image_count} / {image_count_max}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame_right, f'Press "c" to capture', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame_right, f'Press "q" to quit', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame_right, f'RIGHT CAMERA', (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Camera_Right', frame_right)

    detection_status_left = "Checkerboard Detected" if found_left else "No Checkerboard"
    cv2.putText(frame_left, detection_status_left, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame_left, f'Images Captured: {image_count} / {image_count_max}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame_left, f'Press "c" to capture', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame_left, f'Press "q" to quit', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame_left, f'LEFT CAMERA', (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Camera_Left', frame_left)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # If 'c' is pressed, capture the image
    if key == ord('c') and found_right and found_left:
        image_count += 1
        image_path_right = os.path.join(save_directory_right, f'image_{image_count}.jpg')
        cv2.imwrite(image_path_right, frame_right_original)
        print(f'Saved {image_path_right}')

        image_path_left = os.path.join(save_directory_left, f'image_{image_count}.jpg')
        cv2.imwrite(image_path_left, frame_left_original)
        print(f'Saved {image_path_left}')

    elif key == ord('c'):
        print("Checkerboard not detected, image not saved.")

    # If 'q' or 'esc' is pressed, or 30 images have been grabbed quit the loop
    elif key == ord('q') or key == 27 or image_count==image_count_max:
        print("Quitting...")
        break

# Release the camera and close any OpenCV windows
camera_right.release()
camera_left.release()
cv2.destroyAllWindows()
