import cv2
from lane_detection import lane
import piracer


if __name__ == '__main__':
  vehicle = piracer.vehicles.PiRacerStandard()
  cap = cv2.VideoCapture(0)

  cap.set(cv2.CAP_PROP_FPS, 20)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

  while(cap.isOpened()):
    ret, original_frame = cap.read()
    if ret == False:
      break
    frame, radian = lane.lane_tracker(original_frame)

    out.write(frame)
    cv2.imshow('frame', frame)
    piracer.operate_radian.set_steering_radians(piracer=vehicle ,radians=radian)
    if cv2.waitKey(10) == 27:
      break

  cap.release()
  out.release()
  cv2.destroyAllWindows()
