import cv2
import numpy as np
import glob
import os

# Define the dimensions of the checkerboard
CHECKERBOARD = (7, 9)  # (number of internal corners per a chessboard row, number of internal corners per a chessboard column)
square_size = 1  # This is the real world size of a square in your checkerboard (e.g., 1.0 if squares are 1x1 units)

# Directory containing checkerboard images
image_directory = 'calibration_images\stereo_left'

# Termination criteria for the iterative algorithm used to refine the corner positions
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * square_size  # Scale object points by the real size of a square

# Arrays to store object points and image points from all the images
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane



# Fetch all image files in the directory
images = glob.glob(os.path.join(image_directory, '*.jpg'))

# Process each image
for image_file in images:
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Perform camera calibration to obtain camera matrix, distortion coefficients, etc.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save the camera calibration results
calibration_data = {
    'camera_matrix': mtx,
    'dist_coeff': dist,
    'rvecs': rvecs,
    'tvecs': tvecs
}

np.savez('calibration_files\camera_calibration_data.npz', **calibration_data)

print("Calibration successful.")
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)

# Optionally, you can undistort an image to check the calibration result
# img = cv2.imread('some_test_image.jpg')
# h, w = img.shape[:2]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imshow('undistorted', dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
