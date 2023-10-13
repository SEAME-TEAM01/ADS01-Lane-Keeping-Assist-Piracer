# - import python library
import os
import cv2
import time

# - import color variables
from util.color import *

# - variables
VIDEO = 0
TERM_SIZE = os.get_terminal_size().columns

# - capture img program
def capture_video(_):
    # Infor program start
    print(
        f"{CYA}{BOL}[INFORMT]{RES}    ",
        f"Capture-img process has been started at:",
        "\n",
        f"{CYA}{BOL}         {RES}    ",
        f"{time.time()}"
    )

    # Start video capture
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    
    # Verify that the camera is available
    if not cap.isOpened():
        print(
            f"{RED}{BOL}[FAILURE]{RES}    ",
            f"Can't open camera. Please check the connection!"
        )
        return
    
    try:
        while True:
            rst, frame = cap.read()
            frame = cv2.flip(frame, 0)

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
            
            cv2.imwrite(f'dataset/frames/frame_{time.time()}.jpg', frame)
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
