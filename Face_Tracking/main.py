# main menu


def op0_Start_Modbus_Server():
    print('Running: 0_Start_Modbus_Server')

    print(f'Srtarting Modbus')  # Press Ctrl+F8 to toggle the breakpoint.
    # server = ModbusServer("127.0.0.1", 12345, no_block=True)
    print("Start server...")
    # server.start()
    print("Server is online")

def op1_Calibrate_Right_Camera():
    print("Running: 1_Calibrate_Right_Camera")

    #grab images


    #perform Calibration

    #Show calibration results

def op2_Calibrate_Left_Camera():
    print('Running: 2_Calibrate_Left_Camera')

def op3_Calibrate_Stereovision():
    print("Running: 3_Calibrate_Stereovision")

def op4_Single_Camera_Face_Detect():
    print("Running: 4_Run_Single_Camera_Face_Detect ")

def op5_Stereo_Face_Detect():
    print("Running: 5_Stereo_Face_Detect")

    #import all data
    

def op10_Exit():
    print("Running: Exiting Program")



## Start of the program
# This part should only be modified to add more options
# please create a new function and add intructions in there
if __name__ == '__main__':

    while True:
        print('MENU')
        print('     - 0_Start_Modbus_Server')
        print('     - 1_Calibrate_Right_Camera')
        print('     - 2_Calibrate_Left_Camera')
        print('     - 3_Calibrate_Stereovision')
        print('     - 4_Single_Camera_Face_Detect')
        print('     - 5_Stereo_Face_Detect')
        print('     - 10_Exit')

        prog_select = input("What program do you want to run ? ")

        if prog_select == '0':
            op0_Start_Modbus_Server()
        elif prog_select == '1':
            op1_Calibrate_Right_Camera()
        elif prog_select == '2':
            op2_Calibrate_Left_Camera()
        elif prog_select == '3':
            op3_Calibrate_Stereovision()
        elif prog_select == '4':
            op4_Single_Camera_Face_Detect()
        elif prog_select == '5':
            op5_Stereo_Face_Detect()
        elif prog_select == '10':
            op10_Exit()
            break
        else:
            print("NOT AN OPTION")


    #scc.grab_calibration_images(0,15, 10, 7)
    #stereocc.grab_calibration_images(0, 1, 15, 10, 7)
    #stevie.full_code()






