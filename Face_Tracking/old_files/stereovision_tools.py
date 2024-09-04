import math
import numpy as np
import cv2
from facetrack_scripts import face_detect_stereo_v0 as fds0


def stereovision_find_depth(point_left,point_right, img_left, img_right,f, alpha, baseline ):
    # point_left -> pixel coordinates in the left image frame
    # point_right -> pixel coordinates in the right image frame
    # img_left -> left frame image
    # img_right -> right frame image
    # f cameras lense's focal lentgh [mm]  (logitech c170 2.3mm)
    # alpha -> cameras field of view in the horitzontal plane [degrees] (logitech c170 58º)
    # baseline -> distance between cameras [cm]  (home setup 19 cm)
    #

    focal_pixels=0
    height_right, width_right = img_right.shape[:2]
    height_left, width_left = img_left.shape[:2]

    if (height_right == height_left) and (width_right == width_left):
        # convert focal length f from [mm] to [px]
        focal_pixels=(width_right*0.5)/math.tan(alpha*0.5*np.pi/180)
    else:
        print('images have different dimensions')

    #calculate disparity
    disparity=point_left[0]-point_right[0]
    #calculate the depth of z
    z_depth=(baseline - focal_pixels )/disparity  #depth in [cm]

    return abs(z_depth)


# ------------- Setup and Inititalisation ---------------------------------------------------


# Camera setup
cam1 = cv2.VideoCapture(0)  # webcam
cam2 = cv2.VideoCapture(1)  # puerto

#camera parameters
cam1_alpha=58 #[degrees]
cam2_alpha=58 #[degrees]
cam1_focal=2.3 #[mm]
cam2_focal=2.3 #[mm]
cam_baseline=13 #[cm] distance between cameras
cam_depth_multiplier=15.93 #found manually

#face detection initialisation

mp_face_mesh_r, face_mesh_r = fds0.face_initialisation()
mp_face_mesh_l, face_mesh_l= fds0.face_initialisation()

#initial values

count =-1

while cam1.isOpened() and cam2.isOpened():

    count += 1

    success_r, img_r_original = cam1.read()  # adquire frame
    success_l, img_l_original = cam2.read()  # adquire frame

    ######### Camera Calibration ############################################################################

    img_r_undistorted = np.copy(img_r_original)
    img_l_undistorted = np.copy(img_l_original)
    #img_r_undistorted, img_l_undistorted = calib.undistorted(img_r_original, img_l_original)

    ########################################################################################################

    if success_r == True:
        status_r, nose_2d_r = fds0.check_face_orientation(1, img_r_undistorted, face_mesh_r,mp_face_mesh_r)
    if success_l == True:
        status_l, nose_2d_l = fds0.check_face_orientation(2, img_l_undistorted, face_mesh_l, mp_face_mesh_l)

    if (status_r == True) and (status_l == True):
        depth_face=stereovision_find_depth(nose_2d_l, nose_2d_r, img_l_undistorted, img_r_undistorted, cam1_focal, cam1_alpha, cam_baseline)
        depth_face=depth_face*cam_depth_multiplier
        print(f'nose position: {depth_face}')

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam1.release()
cam2.release()
cv2.destroyAllWindows()

