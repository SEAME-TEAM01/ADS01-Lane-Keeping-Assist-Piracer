import cv2
from lane_detection.lane import Lane
import piracer


if __name__ == '__main__':
  vehicle = piracer.vehicles.PiRacerStandard()
  lane = Lane();
  cap = cv2.VideoCapture('assets/piracer_pov.mp4')

  cap.set(cv2.CAP_PROP_FPS, 20)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

  # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  # out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

  i = 0
  while(i < 400):
    ret, original_frame = cap.read()
    if ret == False:
      break
    radian, frame = lane.lane_tracker(original_frame)

    # out.write(frame)
    cv2.imshow('frame', frame)
    # print(i, radian)
    piracer.operate_radian.set_steering_radians(piracer=vehicle ,radians=radian)
    if cv2.waitKey(10) == 27:
      break
    i += 1

  cap.release()
  # out.release()
  cv2.destroyAllWindows()
