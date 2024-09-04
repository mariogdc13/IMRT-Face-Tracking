import cv2
import glob
import numpy as np

####### Setup ###############

#pattern characteristics definitions
PATTERN_SIZE = (10 , 7) # vertices of checkerboard pattern
SQUARE_SIZE=24 # size of checkerboard pattern (mm)

#stereo image directories
directory_right_images = 'calibration_images\stereo_right\*.jpg'
directory_left_images = 'calibration_images\stereo_left\*.jpg'

#Calibration info intrinsic parameters
directory_right_calibration = 'calibration_files\camera_right_calibration_data.npz'
directory_left_calibration = 'calibration_files\camera_left_calibration_data.npz'
directory_stereo_calibration = 'calibration_files\camera_stereo_calibration_data.npz'
directory_stereo_rectification = 'calibration_files\camera_stereo_rectification_data.npz'

############## Checkerboard pattern extraction ##################

#checkerboard definitions
checkerboard_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
left_pts, right_pts = [], []
img_size = None

# load checkerboard images from directory
left_imgs = list(sorted(glob.glob(directory_left_images)))
right_imgs = list(sorted(glob.glob(directory_right_images)))
assert len(left_imgs) == len(right_imgs) # check there are the same number of images

# Extract checkerboard partern from each of the loaded images
for left_img_path, right_img_path in zip(left_imgs, right_imgs):
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
    if img_size is None:
        img_size = (left_img.shape[1], left_img.shape[0])

    res_left, corners_left = cv2.findChessboardCorners(left_img, PATTERN_SIZE)
    res_right, corners_right = cv2.findChessboardCorners(right_img, PATTERN_SIZE)

    corners_left = cv2.cornerSubPix(left_img, corners_left, (11, 11), (-1, -1), checkerboard_criteria)
    corners_right = cv2.cornerSubPix(right_img, corners_right, (11, 11), (-1, -1), checkerboard_criteria)

    left_pts.append(corners_left)
    right_pts.append(corners_right)

pattern_points = np.zeros((np.prod(PATTERN_SIZE), 3), np.float32)
pattern_points[:, :2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2)
pattern_points = [pattern_points * SQUARE_SIZE] * len(left_imgs)
#pattern_points = [pattern_points] * SQUARE_SIZE

####################### STEREO CALIBATION #################################

# load calibration data for intrinsec parameters
calibration_data_left = np.load(directory_left_calibration)
calibration_data_right = np.load(directory_right_calibration)

int_matrix_left = calibration_data_left['camera_matrix']
int_dist_left = calibration_data_left['dist_coeff']
int_matrix_right = calibration_data_right['camera_matrix']
int_dist_right = calibration_data_right['dist_coeff']

#stereo definitions

calibration_flags = 0
#calibration_flags = cv2.CALIB_FIX_INTRINSIC
calibration_flags = cv2.CALIB_USE_INTRINSIC_GUESS
#calibration_flags =  cv2.CALIB_SAME_FOCAL_LENGTH
#calibration_flags = cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_SAME_FOCAL_LENGTH
#calibration_flags = cv2.CALIB_SAME_FOCAL_LENGTH + cv2.CALIB_ZERO_TANGENT_DIST
#calibration_flags = cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_SAME_FOCAL_LENGTH + cv2.CALIB_ZERO_TANGENT_DIST
stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

retStereo, new_int_matrix_left, new_int_dist_left, new_int_matrix_right, new_int_dist_right, R, T, E, F = cv2.stereoCalibrate( pattern_points, left_pts, right_pts, int_matrix_left, int_dist_left, int_matrix_right, int_dist_right, img_size, flags=calibration_flags, criteria=stereo_criteria)
#err, Kl, Dl, Kr, Dr, R, T, E, F = cv2.stereoCalibrate( pattern_points, left_pts, right_pts, None, None, None, None, img_size, flags=cv2.CALIB_SAME_FOCAL_LENGTH + cv2.CALIB_ZERO_TANGENT_DIST)
print('Left camera:')
print(new_int_matrix_left)
print('Left camera distortion:')
print(new_int_dist_left)
print('Right camera:')
print(new_int_matrix_right)
print('Right camera distortion:')
print(new_int_dist_right)
print('Rotation matrix:')
print(R)
print('Translation:')
print(T)
print('Essencial matrix:')
print(E)
print('Fundamental matrix:')
print(F)

stereo_calibration_data = {
    'Kl': new_int_matrix_left,
    'Dl': new_int_dist_left,
    'Kr': new_int_matrix_right,
    'Dr': new_int_dist_right,
    'R': R,
    'T': T,
    'E': E,
    'F': F,
    'img_size': img_size,
    'left_pts': left_pts,
    'right_pts': right_pts
}


np.savez(directory_stereo_calibration, **stereo_calibration_data )


######## Stereo Rectification ################

rectify_scale=0
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv2.stereoRectify(new_int_matrix_left,new_int_dist_left,new_int_matrix_right,new_int_dist_right, img_size, R, T, rectify_scale, (0,0) )

stereoMapL = cv2.initUndistortRectifyMap(new_int_matrix_left, new_int_dist_left, rectL, projMatrixL, img_size,cv2.CV_16SC2)
stereoMapR = cv2.initUndistortRectifyMap(new_int_matrix_right, new_int_dist_right, rectR, projMatrixR, img_size,cv2.CV_16SC2)

stereo_rectification_data = {
    'rectification_L' : rectL,
    'rectification_R' : rectR,
    'projMatrixL': projMatrixL,
    'projMatrixR': projMatrixR,
    'Q':Q,
    'roi_l': roi_L,
    'roi_r': roi_R,
    'stereoMapL_x': stereoMapL[0],
    'stereoMapL_y': stereoMapL[1],
    'stereoMapR_x': stereoMapR[0],
    'stereoMapR_y': stereoMapR[1]
}


np.savez(directory_stereo_rectification, **stereo_rectification_data )