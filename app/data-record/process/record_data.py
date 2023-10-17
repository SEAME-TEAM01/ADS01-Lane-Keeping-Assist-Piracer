# - import python library
import os
import cv2
import time

# - import color variables
from util.color import *

# - variables
VIDEO = 0
CSV_FILE = "dataset/record.csv"
TERM_SIZE = os.get_terminal_size().columns

# - capture img program
def record_data(vehicle):
    # Infor program start
    print(
        f"{CYA}{BOL}[INFORMT]{RES}    ",
        f"Capture-img process has been started at:",
        "\n",
        f"{CYA}{BOL}         {RES}    ",
        f"{time.time()}"
    )

    # file check
    if  not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w") as file:
            file.write("index,steering\n")

    # Start video capture
    cap = cv2.VideoCapture(VIDEO)
    
    # Verify that the camera is available
    if not cap.isOpened():
        print(
            f"{RED}{BOL}[FAILURE]{RES}    ",
            f"Can't open camera. Please check the connection!"
        )
        return
    
    try:
        index = 0
        while True:
            rst, frame  = cap.read()
            frame       = cv2.flip(frame, -1)
            steering    = vehicle.get_steering_raw_data()

            if not rst:
                print(
                    f"{RED}{BOL}[FAILURE]{RES}    ",
                    f"Can't read the frame.",
                    "\n",
                    f"{RED}{BOL}         {RES}    ",
                    f"Please follow steps to solve the problem.",
                    "\n",
                    f"{RED}{BOL}         {RES}    ",
                    f" - Check camera connection and drivers.",
                    "\n",
                    f"{RED}{BOL}         {RES}    ",
                    f" - Check is another application using camera."
                )
                break
            
            cv2.imwrite(f'dataset/frames/frame_{index}_{steering}.jpg', frame)
            with open(CSV_FILE, "a") as csv_file:
                csv_file.write(f"{index},{steering}\n")
            index += 1

    except Exception as exception:
        print(
            f"{RED}{BOL}[FAILURE]{RES}    ",
            f"Unexpected exception has occured.\n",
            '-'*TERM_SIZE, "\n",
            exception, "\n",
            '-'*TERM_SIZE,
        )
    finally:
        cap.release()
        cv2.destroyAllWindows()
