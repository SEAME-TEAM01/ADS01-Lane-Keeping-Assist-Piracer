# - import python library
import os
import time
import traceback

# - import color variables
from util.color import *

# - variables
CSVFILE = "dataset/record.csv"
TERM_SIZE = os.get_terminal_size().columns
THROTTLE_PARAM = 1.0
THROTTLE_INIT = 0.0
STEERING_PARAM = 1.6
STEERING_INIT = -0.2

# - automat run program
def run_action_order(vehicle, throttle, steering):
    vehicle.set_throttle_percent(throttle * THROTTLE_PARAM)
    vehicle.set_steering_percent(steering * STEERING_PARAM)
    with open(CSVFILE, "a") as csv_file:
        csv_file.write(f"{time.time()},{steering},{throttle}\n")


def run_action(vehicle, throttle, steering, duration):
    if  throttle > 1.0 or throttle < -1.0 or\
        steering > 1.0 or steering < -1.0:
        print(
            f"{RED}{BOL}[FAILURE]{RES}    ",
            f"Check the throttle and steering value.",
            "\n",
            f"{RED}{BOL}         {RES}    ",
            f"It must between -1.0 ~ 1.0.",
            "\n",
            f"{RED}{BOL}         {RES}    ",
            f"{YEL}throttle: [{throttle}], steering: [{steering}]{RES}"
        )
        return
    start_time = time.time()
    while time.time() - start_time < duration:
        run_action_order(vehicle, throttle, -steering)
    vehicle.set_steering_percent(STEERING_INIT)

def automat_run(vehicle):
    # Infor program start
    print(
        f"{CYA}{BOL}[INFORMT]{RES}    ",
        f"Automat-run process has been started at:",
        "\n",
        f"{CYA}{BOL}         {RES}    ",
        f"{time.time()}"
    )

    # Initialize objects
    vehicle.set_throttle_percent(THROTTLE_INIT)
    vehicle.set_steering_percent(STEERING_INIT)

    # Check csv label
    if  not os.path.exists(CSVFILE):
        with open(CSVFILE, "w") as file:
            file.write("miliseconds,steering,throttle\n")

    try:
        run_action(
            vehicle,
            throttle=0.3,
            steering=STEERING_INIT,
            duration=2.45
        )
        while True:
            run_action(
                vehicle,
                throttle=0.3,
                steering=0.55,
                duration=4.65
            )
            run_action(
                vehicle,
                throttle=0.3,
                steering=STEERING_INIT,
                duration=1.6
            )
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
        vehicle.set_steering_percent(THROTTLE_INIT)
        vehicle.set_throttle_percent(STEERING_INIT)