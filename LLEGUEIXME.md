Aquest repositori conté els codis de Pyhton, les imatges de calibració i els arxius de calibració necessaris per realitzar un projecte d'INTEGRACIÓ DE SEGUIMENT FACIAL EN TRACTAMENTS DE RADIOTERÀPIA D’INTENSITAT MODULADA.
-
L'ordre d'execució dels codis és el següent:
1 -> 11_grab_30_images_left.py
2 -> 12_grab_30_images_right.py
3 -> 13_grab_30_images_stereo.py
4 -> 21_pinhole_calibration_left_from_file.py
5 -> 22_pinhole_calibration_right_from_file.py
6 -> 23_3_stereo_camera_calibration_from_file.py
7 -> 31_2_track_face_position_stereo.py
8 -> 40_modbus.py
-

Per poder executar els codis es necessari instal·lar prèviament les següents llibreries:
* cv2
* numpy
* glob
* os
* mediapipe
* time
* math
* pyModbusTCP.server
