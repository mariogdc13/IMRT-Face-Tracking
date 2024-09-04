import cv2
import numpy as np

# Load the calibration data from the *.npz file
calibration_data = np.load('calibration_files\camera_right_calibration_data.npz')

# Extract the camera matrix and distortion coefficients
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeff']

# Initialize the camera (default camera is usually index 0)
camera = cv2.VideoCapture(0) # right camera
#camera = cv2.VideoCapture(1) #left camera

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()



while True:
    # Capture frame-by-frame
    ret, frame = camera.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Display the frame
    cv2.imshow('Camera raw', frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF


    # Get the image dimensions
    h, w = frame.shape[:2]

    # Get the optimal camera matrix for better undistortion
    optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # Undistort the image
    undistorted_image = cv2.undistort(frame, camera_matrix, dist_coeffs, None, optimal_camera_matrix)

    # Crop the image based on the region of interest (roi) to remove black borders
    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]

    # Save or display the undistorted image
    #cv2.imwrite('undistorted_image.jpg', undistorted_image)
    cv2.imshow('Undistorted Image', undistorted_image)

    # If 'q' or 'Esc' is pressed, quit the loop
    if key == ord('q') or key == 27:  # 27 is the ASCII code for 'Esc'
        print("Quitting...")
        break

# Release the camera and close any OpenCV windows
camera.release()
cv2.destroyAllWindows()
