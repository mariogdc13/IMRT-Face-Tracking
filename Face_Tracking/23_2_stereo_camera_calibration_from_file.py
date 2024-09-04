import cv2
import glob
import numpy as np

PATTERN_SIZE = (7 , 4)
SQUARE_SIZE=34
left_imgs = list(sorted(glob.glob('calibration_images\stereo_left\*.jpg')))
right_imgs = list(sorted(glob.glob('calibration_images\stereo_right\*.jpg')))
assert len(left_imgs) == len(right_imgs)

# load calibration data
calibration_data_left = np.load('calibration_files\camera_left_calibration_data.npz')
calibration_data_right = np.load('calibration_files\camera_right_calibration_data.npz')

camera_matrix_right = calibration_data_right['camera_matrix']
camera_distortion_right = calibration_data_right['dist_coeff']
camera_matrix_left = calibration_data_left['camera_matrix']
camera_distortion_left = calibration_data_left['dist_coeff']

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
left_pts, right_pts = [], []
img_size = None

for left_img_path, right_img_path in zip(left_imgs, right_imgs):
    left_img = cv2.imread(left_img_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_img_path, cv2.IMREAD_GRAYSCALE)
    if img_size is None:
        img_size = (left_img.shape[1], left_img.shape[0])

    res_left, corners_left = cv2.findChessboardCorners(left_img, PATTERN_SIZE)
    res_right, corners_right = cv2.findChessboardCorners(right_img, PATTERN_SIZE)

    corners_left = cv2.cornerSubPix(left_img, corners_left, (11, 11), (-1, -1), criteria)
    corners_right = cv2.cornerSubPix(right_img, corners_right, (11, 11), (-1, -1), criteria)

    left_pts.append(corners_left)
    right_pts.append(corners_right)

#
pattern_points = np.zeros((np.prod(PATTERN_SIZE), 3), np.float32)
pattern_points[:, :2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2)
pattern_points = [pattern_points * SQUARE_SIZE] * len(left_imgs)
#pattern_points = [pattern_points] * SQUARE_SIZE

#calibration_flags = 0
#calibration_flags = cv2.CALIB_FIX_INTRINSIC
calibration_flags =  cv2.CALIB_SAME_FOCAL_LENGTH
#calibration_flags = cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_SAME_FOCAL_LENGTH
#calibration_flags = cv2.CALIB_SAME_FOCAL_LENGTH + cv2.CALIB_ZERO_TANGENT_DIST
#calibration_flags = cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_SAME_FOCAL_LENGTH + cv2.CALIB_ZERO_TANGENT_DIST
stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

err, Kl, Dl, Kr, Dr, R, T, E, F = cv2.stereoCalibrate( pattern_points, left_pts, right_pts, camera_matrix_left, camera_distortion_left, camera_matrix_right, camera_distortion_right, img_size, flags=calibration_flags, criteria=stereo_criteria)
#err, Kl, Dl, Kr, Dr, R, T, E, F = cv2.stereoCalibrate( pattern_points, left_pts, right_pts, None, None, None, None, img_size, flags=cv2.CALIB_SAME_FOCAL_LENGTH + cv2.CALIB_ZERO_TANGENT_DIST)
print('Left camera:')
print(Kl)
print('Left camera distortion:')
print(Dl)
print('Right camera:')
print(Kr)
print('Right camera distortion:')
print(Dr)
print('Rotation matrix:')
print(R)
print('Translation:')
print(T)
print('Essencial matrix:')
print(E)
print('Fundamental matrix:')
print(F)

calibration_data = {
    'Kl': Kl,
    'Dl': Dl,
    'Kr': Kr,
    'Dr': Dr,
    'R': R,
    'T': T,
    'E': E,
    'F': F,
    'img_size': img_size,
    'left_pts': left_pts,
    'right_pts': right_pts
}


np.savez('calibration_files\camera_stereo_calibration_data.npz', **calibration_data )
