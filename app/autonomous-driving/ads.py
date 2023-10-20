import  cv2
import  numpy
from    tensorflow.keras.models import load_model
from    piracer.vehicles import PiRacerStandard
from    piracer.gamepads import ShanWanGamepad

WIDTH = 640
HEIGHT = 480
MODEL = "model.h5"
THROTTLE_PARAM = 1.0
THROTTLE_INIT  = 0.0
STEERING_PARAM = -1.5
STEERING_INIT  = 0.0

def preprocessing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # RGB to Gray
    image = cv2.resize(image, (WIDTH, HEIGHT)) / 255.0
    image = numpy.expand_dims(image, axis=-1) # add dimensions (HEIGHT, WIDTH) -> (HEIGHT, WIDTH, 1)
    return image

if  __name__ == "__main__":
    mod = load_model(MODEL)
    cap = cv2.VideoCapture(0)

    try:
        vehicle = PiRacerStandard()
        gamepad = ShanWanGamepad()
        vehicle.set_steering_percent(STEERING_INIT)
        vehicle.set_throttle_percent(THROTTLE_INIT)

        while True:
            _, frame = cap.read()
            if not _:
                print("Failed to grab frame")
                break

            frame = preprocessing(frame)
            frame_batch = numpy.expand_dims(frame, axis=0)

            predc = mod.predict(frame_batch)
            predc_label = numpy.argmax(predc, axis=1)[0]

            steering = STEERING_INIT
            if predc_label == 0:
                steering = STEERING_INIT
            elif predc_label == 1:
                steering = -0.57
            elif predc_label == 2:
                steering = 0.76

            throttle = 0.3

            vehicle.set_steering_percent(steering)
            vehicle.set_throttle_percent(throttle)
    except Exception as exception:
        print("Unexpected error was occured.", exception)
    finally:
        vehicle.set_steering_percent(THROTTLE_INIT)
        vehicle.set_throttle_percent(STEERING_INIT)
