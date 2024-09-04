#This is a python script to calibrate a single camera

import cv2
import numpy as np


def grab_calibration_images(camera_port_r, camera_port_l,n_calibration_images, n_columns, n_rows):

    print('Start calibration: image grabbing')

    checkerboard_dim=(n_columns, n_rows)
    cnt_grabbed_images=0

    # stop the iteration when specified
    # accuracy, epsilon, is reached or
    # specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    threedpoints_r = [] # Vector for 3D points
    twodpoints_r = [] # Vector for 2D points

    threedpoints_l = []  # Vector for 3D points
    twodpoints_l = []  # Vector for 2D points

    #  3D points real world coordinates
    objectp3d_r = np.zeros((1, checkerboard_dim[0] * checkerboard_dim[1],3), np.float32)
    objectp3d_l = np.zeros((1, checkerboard_dim[0] * checkerboard_dim[1], 3), np.float32)

    objectp3d_r[0, :, :2] = np.mgrid[0:checkerboard_dim[0], 0:checkerboard_dim[1]].T.reshape(-1, 2)
    objectp3d_l[0, :, :2] = np.mgrid[0:checkerboard_dim[0], 0:checkerboard_dim[1]].T.reshape(-1, 2)

    prev_img_shape_r = None
    prev_img_shape_l = None

    cam_object_r = cv2.VideoCapture(camera_port_r)  # webcam
    cam_object_l = cv2.VideoCapture(camera_port_l)  # webcam

    while cam_object_r.isOpened() and cam_object_l.isOpened():
        success_r, img_original_r = cam_object_r.read()  # adquire frame
        success_l, img_original_l = cam_object_l.read()  # adquire frame
        img_original_r = cv2.flip(img_original_r, 1)
        img_original_l = cv2.flip(img_original_l, 1)

        if cv2.waitKey(5) & 0xFF == ord('c'): #if c letter is pressed
            img_overlayed_r = np.copy(img_original_r)
            img_overlayed_l = np.copy(img_original_l)
            img_gray_r = cv2.cvtColor(img_original_r, cv2.COLOR_BGR2GRAY) #turnimage to gray scale
            img_gray_l = cv2.cvtColor(img_original_l, cv2.COLOR_BGR2GRAY)  # turnimage to gray scale

            ret_r, corners_r = cv2.findChessboardCorners(img_gray_r, checkerboard_dim, cv2.CALIB_CB_ADAPTIVE_THRESH  + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret_r==True:
                threedpoints_r.append(objectp3d_r)
                corners2_r = cv2.cornerSubPix(img_gray_r, corners_r, (11, 11), (-1, -1), criteria)
                twodpoints_r.append(corners2_r)

                #show captured image with overlayed chessboard
                img_overlayed = cv2.drawChessboardCorners(img_overlayed, checkerboard_dim, corners2_r, ret_r)
                cv2.imshow(f'obtained calibrations image right', img_overlayed)

                # save original image to file
                cv2.imwrite(f'calibration_images/r/calibration{cnt_grabbed_images}.jpg', img_original_r)
                cnt_grabbed_images=cnt_grabbed_images+1
                print('right image saved')

            ret_l, corners_l = cv2.findChessboardCorners(img_gray_l, checkerboard_dim,
                                                         cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret_l==True:
                threedpoints_l.append(objectp3d_l)
                corners2_l = cv2.cornerSubPix(img_gray_l, corners_l, (11, 11), (-1, -1), criteria)
                twodpoints_l.append(corners2_l)

                #show captured image with overlayed chessboard
                img_overlayed = cv2.drawChessboardCorners(img_overlayed, checkerboard_dim, corners2_l, ret_l)
                cv2.imshow(f'obtained calibrations image left', img_overlayed)

                # save original image to file
                cv2.imwrite(f'calibration_images/l/calibration{cnt_grabbed_images}.jpg', img_original_l)
                cnt_grabbed_images=cnt_grabbed_images+1
                print('left image saved')


        #show information on screen
        img_annotated_r = np.copy(img_original_r)
        cv2.putText(img_annotated_r, f'Images right Acquired: {int(cnt_grabbed_images)}/{int(n_calibration_images)}', (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.imshow(f'Camera right{camera_port_r}', img_annotated_r)

        img_annotated_l = np.copy(img_original_l)
        cv2.putText(img_annotated_l, f'Images Acquired: {int(cnt_grabbed_images)}/{int(n_calibration_images)}', (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.imshow(f'Camera left{camera_port_l}', img_annotated_l)

        if (cv2.waitKey(5) & 0xFF == 27):
            # program is aborted with esc key
            cv2.destroyAllWindows()
            cam_object_r.release()
            cam_object_l.release()
            break

        if cnt_grabbed_images>=n_calibration_images:
            # all "n_calibration_images" images have been captured

            #calculate the caliration
            h, w = img_original_r.shape[:2]
            ret_r, matrix_r, distortion_r, r_vecs_r, t_vecs_r = cv2.calibrateCamera(threedpoints_r, twodpoints_r, img_gray_r.shape[::-1], None, None)
            # Displaying required output
            print(" Camera matrix right:")
            print(matrix_r)
            print("\n Distortion coefficient right:")
            print(distortion_r)
            print("\n Rotation Vectors right:")
            print(r_vecs_r)
            print("\n Translation Vectors right:")
            print(t_vecs_r)

            h, w = img_original_r.shape[:2]
            ret_l, matrix_l, distortion_l, r_vecs_l, t_vecs_l = cv2.calibrateCamera(threedpoints_l, twodpoints_l,
                                                                          img_gray_l.shape[::-1], None, None)
            # Displaying required output
            print(" Camera matrix right:")
            print(matrix_l)
            print("\n Distortion coefficient right:")
            print(distortion_l)
            print("\n Rotation Vectors right:")
            print(r_vecs_l)
            print("\n Translation Vectors right:")
            print(t_vecs_l)

            #saving camera calibration to file

            #reprojection error is best close to zero


            cv2.destroyAllWindows()
            cam_object_r.release()
            cam_object_l.release()
            break

    #image show on screen