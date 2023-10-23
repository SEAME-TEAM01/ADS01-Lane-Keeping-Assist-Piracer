# ------------------------------------------------------------------------------
# Library Import
import  cv2
import  numpy as np
import  tensorflow as tf

from    tensorflow.keras.models \
        import  load_model
from    piracer.vehicles \
        import  PiRacerStandard

# Custom Library Import
from    srcs.colors \
        import  *
from    srcs.variables \
        import  *
from    srcs.preprocess \
        import  preprocessing

# ------------------------------------------------------------------------------
# Run
def run():
    model   = load_model(MODEL)
    capture = cv2.VideoCapture(VIDEO)
    vehicle = PiRacerStandard()

    try:
        vehicle.set_steering_percent(STEERING_INIT)
        vehicle.set_throttle_percent(THROTTLE_INIT)

        while True:
            rst, frame = capture.read()
            if not rst:
                print(
                    f"{RED}{BOL}[FAILURE]{RES}    ",
                    f"Failed to grab frame. Check the camera[{BOL}/dev/video0{RES}]",
                )
                break

            frame = preprocessing(frame, isTest=True)

            predict = model.predict(frame)
            predict_label = np.argmax(predict, axis=1)[0]

            throttle = THROTTLE
            steering = STEERING_INIT
            if predict_label == 0:
                steering = STEERING_INIT
            elif predict_label == 1:
                steering = STEERING_LEFT
            elif predict_label == 2:
                steering = STEERING_RIGHT
            
            vehicle.set_throttle_percent(throttle)
            vehicle.set_steering_percent(steering)

    except Exception as exception:
        print(
            f"{RED}{BOL}[FAILURE]{RES}    ",
            f"Unexpected exception has occured.\n",
            '-'*TERM_SIZE, "\n",
            exception, "\n",
            '-'*TERM_SIZE,
        )

    finally:
        vehicle.set_steering_percent(THROTTLE_INIT)
        vehicle.set_throttle_percent(STEERING_INIT)

# ------------------------------------------------------------------------------
# Main
if  __name__ == "__main__":
    run()