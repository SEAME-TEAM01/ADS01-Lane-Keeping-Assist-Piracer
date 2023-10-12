# - import python library
import os
import sys
from multiprocessing import Process

from piracer.vehicles import PiRacerStandard
from piracer.gamepads import ShanWanGamepad

# - import multiprocess functions
from process import capture_img, control_car, automat_run

# - import color variables
from util.color import *

TERM_SIZE = os.get_terminal_size().columns - 2

def check_cwd():
    """
    Check is current working directory [~/.../app/data-record/]
    """

    folder = "app/data-record"
    cur = os.getcwd()

    if  not cur.endswith(folder):
        print(
            f"{RED}{BOL}[FAILURE]{RES}    ",
            f"You are working on {YEL}wrong directory.{RES}",
            "\n",
            f"{RED}{BOL}         {RES}    ",
            f"Please launch program in [.../{folder}]."
        )
        sys.exit(1)

def multiprocess_create(*args):
    prcs = []
    for arg in args:
        prc = Process(
            name    = arg[0],
            target  = arg[1],
            args    = (arg[2],)
        )
        prcs.append(prc)
    return prcs


# - main
if  __name__ == "__main__":
    check_cwd()
    prcs = []
    vehicle = PiRacerStandard()
    gamepad = ShanWanGamepad()

    try:
        prcs = multiprocess_create(
            ["python3-capture-img", capture_img.capture_img, (1)],
            # ["python3-control-car", control_car.control_car, (vehicle, gamepad)],
            ["python3-automat-run", automat_run.automat_run, (vehicle)],
        )
        print("Debug")
        for prc in prcs:
            prc.start()
        for prc in prcs:
            prc.join()
    except KeyboardInterrupt:
        print(
            f"{CYA}{BOL}[INFORMT]{RES}    ",
            f"Program has been stoped by Keyboard Inturrupt.",
            "\n",
            f"{CYA}{BOL}         {RES}    ",
            f"{GRE}{BOL}GOOD BYE!{RES}"
        )
    except Exception as exception:
        print(
            f"{RED}{BOL}[FAILURE]{RES}    ",
            f"Unexpected exception has occured.\n",
            '-'*TERM_SIZE, "\n",
            exception, "\n",
            '-'*TERM_SIZE,
        )
    finally:
        for prc in prcs:
            prc.terminate()
        vehicle.set_steering_percent(0)
        vehicle.set_throttle_percent(0)