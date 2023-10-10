from        piracer.vehicles    import  PiRacerStandard
from        piracer.gamepads    import  ShanWanGamepad
import      cv2                 as      cv2
import      time                as      tim
import      traceback           as      trc
import      os

CSVFILE     = "steering.csv"

def car_control():
    print("car control started.")
    pad = ShanWanGamepad()
    veh = PiRacerStandard()

    # Start video capture
    cap = cv2.VideoCapture(0)
    
    # Verify that the camera is available
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cnt = 0
    try:
        while True:
            # Capture frame-by-frame
            rst, frame = cap.read()
            frame = cv2.flip(frame, 0)

            # Check if frame captured successfully
            if not rst:
                print("Error: Could not read frame.")
                break
            
            # Increment frame count
            cnt += 1
            
            # Save the frame
            cv2.imwrite(f'frame_{cnt}.jpg', frame)

            gamepad_input   = pad.read_data()
            throttle        = gamepad_input.analog_stick_right.y 
            steering        = gamepad_input.analog_stick_left.x

            veh.set_steering_percent(-steering)
            veh.set_throttle_percent(throttle)

            # Log steering and frame info
            with open(CSVFILE, "a") as csv_file:
                csv_file.write(f"{cnt}, {steering}\n")
            
            # Control the frequency of the loop - this will depend on the frame rate of your camera
            # tim.sleep(0.1)

    except Exception as e:
        print(" - Car control process has been stopped. - ")
        print("An error occurred:", e)     
        trc.print_exc()

        # Reset drivetrain
        veh.set_steering_percent(0)
        veh.set_throttle_percent(0)

    except KeyboardInterrupt:
        print("Car control process has been stopped.")
        
        veh.set_steering_percent(0)
        veh.set_throttle_percent(0)

    finally:
        # Release the video capture object
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    if not os.path.exists(CSVFILE):
        with open(CSVFILE, "w") as file:
            file.write("frame_count,steering\n")
    car_control()
