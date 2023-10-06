import cv2
from lane_detection import lane
import piracer


if __name__ == '__main__':
  vehicle = piracer.vehicles.PiRacerStandard()
  cap = cv2.VideoCapture(0)
  ret, original_frame = cap.read()
  while(cap.isOpened()):
    ret, original_frame = cap.read()
    if ret == False:
      break
    frame, radian = lane.lane_tracker(original_frame)
    cv2.imshow('frame', frame)
    piracer.operate_radian.set_steering_radians(piracer=vehicle ,radians=radian)
    if cv2.waitKey(10) == 27:
      break
  cap.release()
  cv2.destroyAllWindows()
