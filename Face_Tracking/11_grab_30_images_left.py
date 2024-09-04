import cv2
import os

# Set the directory where images will be saved
save_directory = 'calibration_images/single_left'
os.makedirs(save_directory, exist_ok=True)

# Initialize the camera (default camera is usually index 0)
#camera = cv2.VideoCapture(0) # right camera
camera = cv2.VideoCapture(1) #left camera
#camera = cv2.VideoCapture(2)

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

image_count = 0
image_count_max = 30
checkerboard_size = (10, 7)


while True:
    # Capture frame-by-frame
    ret, frame_original = camera.read()
    frame = frame_original.copy()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the checkerboard corners in the frame
    found, corners = cv2.findChessboardCorners(gray_frame, checkerboard_size, None)

    # If a checkerboard is found, draw it on the frame
    if found:
        cv2.drawChessboardCorners(frame, checkerboard_size, corners, found)

    # write information on screen
    detection_status = "Checkerboard Detected" if found else "No Checkerboard"
    cv2.putText(frame, detection_status, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Images Captured: {image_count} / {image_count_max}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Press "c" to capture', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Press "q" to quit', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'LEFT CAMERA', (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Camera', frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # If 'c' is pressed, capture the image
    if key == ord('c'):
        image_count += 1
        image_path = os.path.join(save_directory, f'image_{image_count}.jpg')
        cv2.imwrite(image_path, frame_original)
        print(f'Saved {image_path}')

    elif key == ord('c'):
        print("Checkerboard not detected, image not saved.")

    # If 'q' or 'esc' is pressed, or 30 images have been grabbed quit the loop
    elif key == ord('q') or key == 27 or image_count==image_count_max:
        print("Quitting...")
        break

# Release the camera and close any OpenCV windows
camera.release()
cv2.destroyAllWindows()
