#This is a python script to calibrate a single camera

import cv2
import numpy as np

def grab_calibration_images(camera_port,n_calibration_images, n_columns, n_rows):

    print('Start calibration: image grabbing')

    checkerboard_dim=(n_columns, n_rows)
    cnt_grabbed_images=0

    # stop the iteration when specified
    # accuracy, epsilon, is reached or
    # specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    threedpoints = [] # Vector for 3D points
    twodpoints = [] # Vector for 2D points

    #  3D points real world coordinates
    objectp3d = np.zeros((1, checkerboard_dim[0] * checkerboard_dim[1],3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:checkerboard_dim[0],0:checkerboard_dim[1]].T.reshape(-1, 2)
    prev_img_shape = None

    cam_object = cv2.VideoCapture(camera_port)  # webcam

    while cam_object.isOpened():
        success, img_original = cam_object.read()  # adquire frame
        img_original = cv2.flip(img_original, 1)

        if cv2.waitKey(5) & 0xFF == ord('c'): #if c letter is pressed
            img_overlayed = np.copy(img_original)
            img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY) #turnimage to gray scale
            ret, corners = cv2.findChessboardCorners(img_gray, checkerboard_dim, cv2.CALIB_CB_ADAPTIVE_THRESH  + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret==True:
                threedpoints.append(objectp3d)
                corners2 = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
                twodpoints.append(corners2)

                #show captured image with overlayed chessboard
                img_overlayed = cv2.drawChessboardCorners(img_overlayed, checkerboard_dim, corners2, ret)
                cv2.imshow(f'obtained calibrations image', img_overlayed)

                # save original image to file
                cv2.imwrite(f'calibration_images/calibration{cnt_grabbed_images}.jpg', img_original)
                cnt_grabbed_images=cnt_grabbed_images+1
                print('image saved')

        #show information on screen
        img_annotated = np.copy(img_original)
        cv2.putText(img_annotated, f'Images Acquired: {int(cnt_grabbed_images)}/{int(n_calibration_images)}', (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.imshow(f'Camera{camera_port}', img_annotated)

        if (cv2.waitKey(5) & 0xFF == 27):
            # program is aborted with esc key
            cv2.destroyAllWindows()
            cam_object.release()
            break

        if cnt_grabbed_images>=n_calibration_images:
            # all "n_calibration_images" images have been captured

            #calculate the caliration
            h, w = img_original.shape[:2]
            ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, img_gray.shape[::-1], None, None)
            # Displaying required output
            print(" Camera matrix:")
            print(matrix)
            print("\n Distortion coefficient:")
            print(distortion)
            print("\n Rotation Vectors:")
            print(r_vecs)
            print("\n Translation Vectors:")
            print(t_vecs)

            #saving camera calibration to file

            #reprojection error is best close to zero


            cv2.destroyAllWindows()
            cam_object.release()
            break

    #image show on screen


def calibrate_camera_from_images(camera_name, camera_port, checkboard_size, ):

    #camera -> camera name (usually right o left) it is used to save the camera calibration to a file
    #camera_port -> port where camera is connected (usually 0,1,2)
    #checkboard_size -> chackboard size in the format [n_columns, n_rows]

    print(f'starting camera calibration from images in: */calibration/{camera_name}/*.jpg ')




    samples = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res, corners = cv2.findChessboardCorners(frame, pattern_size)

        img_show = np.copy(frame)
        cv2.drawChessboardCorners(img_show, pattern_size, corners, res)
        cv2.putText(img_show, 'Samples captured: %d' % len(samples), (0, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('chessboard', img_show)

        wait_time = 0 if res else 30
        k = cv2.waitKey(wait_time)

        if k == ord('s') and res:
            samples.append((cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), corners))
        elif k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    for i in range(len(samples)):
        img, corners = samples[i]
        corners = cv2.cornerSubPix(img, corners, (10, 10), (-1, -1), criteria)

    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

    images, corners = zip(*samples)

    pattern_points = [pattern_points] * len(corners)

    rms, camera_matrix, dist_coefs, rvecs, tvecs = \
        cv2.calibrateCamera(pattern_points, corners, images[0].shape, None, None)

    np.save('camera_mat.npy', camera_matrix)
    np.save('dist_coefs.npy', dist_coefs)


def show_camera_distortion():
    print('showing camera distortion')