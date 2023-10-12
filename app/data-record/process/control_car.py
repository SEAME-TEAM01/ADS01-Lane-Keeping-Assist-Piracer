# - import python library
import os
import time
import traceback

# - import color variables
from util.color import *

# - variables
CSVFILE = "dataset/record.csv"
TERM_SIZE = os.get_terminal_size().columns
THROTTLE_PARAM = 0.6
STEERING_PARAM = 1.3

# - control car program
def control_car(vehicle, gamepad):
    # Infor program start
    print(
        f"{CYA}{BOL}[INFORMT]{RES}    ",
        f"Capture-img process has been started at:",
        "\n",
        f"{CYA}{BOL}         {RES}    ",
        f"{time.time()}"
    )

    # Initialize objects
    vehicle.set_steering_percent(0)
    vehicle.set_throttle_percent(0)

    # Check csv label
    if  not os.path.exists(CSVFILE):
        with open(CSVFILE, "w") as file:
            file.write("miliseconds,steering,throttle\n")

    try:
        while True:
            gamepad_input   = gamepad.read_data()
            throttle        = gamepad_input.analog_stick_left.y
            steering        = gamepad_input.analog_stick_right.x

            vehicle.set_throttle_percent(throttle * THROTTLE_PARAM)
            vehicle.set_steering_percent(steering * STEERING_PARAM)

            # Log steering and frame info
            with open(CSVFILE, "a") as csv_file:
                csv_file.write(f"{time.time()},{steering},{throttle}\n")

    except Exception as exception:
        print(
            f"{RED}{BOL}[FAILURE]{RES}    ",
            f"Unexpected exception has occured.\n",
            f"{BOL}", "-"*TERM_SIZE, f"{RES}\n",
            exception,
            "-" * TERM_SIZE,
        )
        print(
            f"{RED}{BOL}[FAILURE]{RES}    ",
            f"Exception log by traceback:\n",
            f"{BOL}", "-" * TERM_SIZE)
        traceback.print_exc()
        print(
            "-" * TERM_SIZE, f"{RES}",
        )
    finally:
        vehicle.set_steering_percent(0)
        vehicle.set_throttle_percent(0)