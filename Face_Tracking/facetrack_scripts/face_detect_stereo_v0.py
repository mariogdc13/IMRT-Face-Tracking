import cv2
import mediapipe as mp
#from Modbus import ModbusClient, ModbusServer
import numpy as np
import time


#-------------- Face Detect Functions ------------------------------------------------------------

def face_initialisation():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5)  # default static_image_mode=False,
    return mp_face_mesh,face_mesh
def check_face_orientation(camera_n, image, face_mesh, mp_face_mesh):

    processing_start_time = time.time()  # start timer

    image = cv2.flip(image, 1) # Flip the image horizontally for a later selfie-view display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert the color space from BGR to RGB
    image.flags.writeable = False  # To improve performance
    results = face_mesh.process(image)  # Get the result
    image.flags.writeable = True # To improve performance
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # convert the color space from RGB back to BGR


    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks: #if Face Has been Found

        for face_landmarks in results.multi_face_landmarks: #if several faces are present

            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 1 or idx == 33 or idx == 61 or idx == 199 or idx == 263 or idx == 291 :
                    if idx == 1: #if nose has been located
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)



        processing_end_time = time.time()
        annotated_image=draw_nose_vector_on_image(image, nose_2d, x, y) #draw nose vector on image
        annotated_image=draw_landmarks_on_image(annotated_image, face_landmarks, mp_face_mesh) #draw detected features on image
        screen_print_fps(camera_n, image, processing_start_time, processing_end_time) # print the FPS on image
        #screen_print_orientation(camera_n,annotated_image, x, y, z) # print the rx,ry,rz on the image
        screen_print_orientation(camera_n, annotated_image, nose_3d[0], nose_3d[1], nose_3d[2])
        show_data_on_screen(camera_n, annotated_image) #print image on screen

        return True, nose_2d # a face has been found

    else:
        cv2.putText(image, f'NO FACE FOUND', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.imshow(f'Head Pose Estimation cam{camera_n}', image)
        return False, [0,0]# no face has been found
def draw_nose_vector_on_image(rgb_image, nose_2d, x, y):
    annotated_image = np.copy(rgb_image)

    p1 = (int(nose_2d[0]), int(nose_2d[1]))
    p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

    cv2.line(annotated_image, p1, p2, (255, 0, 0), 3)

    return annotated_image
def draw_landmarks_on_image(rgb_image, detection_result, mp_face_mesh):
    annotated_image = np.copy(rgb_image)

    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=[224, 224, 224], thickness=1, circle_radius=1) #white
    drawing_spec_left_eye = mp.solutions.drawing_utils.DrawingSpec(color=[0, 128, 0], thickness=1, circle_radius=1) #green
    drawing_spec_left_eyebrow = mp.solutions.drawing_utils.DrawingSpec(color=[0, 128, 0], thickness=1, circle_radius=1) #green
    drawing_spec_right_eye = mp.solutions.drawing_utils.DrawingSpec(color=[0, 0, 255], thickness=1, circle_radius=1) #red
    drawing_spec_right_eyebrow = mp.solutions.drawing_utils.DrawingSpec(color=[0, 0, 255], thickness=1, circle_radius=1) #red
    drawing_spec_mouth = mp.solutions.drawing_utils.DrawingSpec(color=[255, 0, 0], thickness=1, circle_radius=1) #blue
    drawing_spec_nose = mp.solutions.drawing_utils.DrawingSpec(color=[255, 0, 0], thickness=1, circle_radius=1) #blue
    drawing_spec_oval = mp.solutions.drawing_utils.DrawingSpec(color=[224, 224, 224], thickness=1, circle_radius=1) #white

    #is_drawing_landmarks =True to show all face landmarks - elese False
    # uncoment block to show face tesselation
    '''     
    mp.solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=detection_result,
        connections= mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec,
        is_drawing_landmarks=False)
    '''

    mp.solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=detection_result,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec,
        is_drawing_landmarks=False)

    mp.solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=detection_result,
        connections=mp_face_mesh.FACEMESH_NOSE,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec_nose,
        is_drawing_landmarks=False)

    mp.solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=detection_result,
        connections=mp_face_mesh.FACEMESH_LEFT_EYEBROW,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec_left_eyebrow,
        is_drawing_landmarks=False)

    mp.solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=detection_result,
        connections=mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec_right_eyebrow,
        is_drawing_landmarks=False)

    mp.solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=detection_result,
        connections=mp_face_mesh.FACEMESH_LEFT_EYE,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec_left_eye,
        is_drawing_landmarks=False)

    mp.solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=detection_result,
        connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec_right_eye,
        is_drawing_landmarks=False)

    mp.solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=detection_result,
        connections=mp_face_mesh.FACEMESH_LIPS,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec_mouth,
        is_drawing_landmarks=False)


    return annotated_image
def screen_print_orientation(cam_n, image, face_rx, face_ry, face_rz):
    # See where the user's head tilting
    if face_ry < -10:
        text = "Looking Left"
    elif face_ry > 10:
        text = "Looking Right"
    elif face_rx < -10:
        text = "Looking Down"
    elif face_rx > 10:
        text = "Looking Up"
    else:
        text = "Forward"

    # Add the text on the image
    cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.putText(image, "x: " + str(np.round(face_rx, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, "y: " + str(np.round(face_ry, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, "z: " + str(np.round(face_rz, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
def screen_print_fps(cam_n, image, start_time, end_time):

    elapsed_processing_time = end_time - start_time

    if elapsed_processing_time != 0:
        fps = 1 / elapsed_processing_time
    else:
        fps = 0

    cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
def show_data_on_screen(cam_n, image):
    cv2.imshow(f'Head Pose Estimation cam{cam_n}', image)

#-------------- Modbus Functions ------------------------------------------------------------

def modbus_initialise_communication(ip_adress):
    print(f'Initialising Modbus Communication on ip adress, {ip_adress}')
def modbus_send_data(send_x,send_y,send_z,send_rx,send_ry,send_rz):
    print(f'Sending {send_x,send_y,send_z,send_rx,send_ry,send_rz}')

# ------------- Setup and Inititalisation ---------------------------------------------------

def full_code():
    # Camera setup
    cam1 = cv2.VideoCapture(1)  # webcam
    cam2 = cv2.VideoCapture(0)  # puerto

    # Face_Detection setup
    mp_face_mesh1, face_mesh1 = face_initialisation()
    mp_face_mesh2, face_mesh2 = face_initialisation()


    while cam1.isOpened() and cam2.isOpened():

        success1, image1 = cam1.read()  # adquire frame
        success2, image2 = cam2.read()  # adquire frame

        if success1 == True:
            check_face_orientation(1, image1,face_mesh1,mp_face_mesh1)
        if success2 == True:
            check_face_orientation(2, image2,face_mesh2,mp_face_mesh2)

        '''
        # show image data on screen
        if success1 == True:
            show_data_on_screen(1, image1)
        if success2 == True:
            show_data_on_screen(2,image2)
        '''

        #print info on screen

        #send data via modbus
        send_x = 10
        send_y= 10
        send_z= 10
        send_rx = 10
        send_ry = 10
        send_rz = 10

        #if success1 == True and success2 == True:
        #    modbus_send_data(send_x, send_y, send_z, send_rx, send_ry, send_rz)





        if cv2.waitKey(5) & 0xFF == 27:
            break

    cam1.release()
    cam2.release()

#main

full_code()



