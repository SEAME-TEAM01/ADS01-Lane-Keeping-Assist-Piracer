import cv2
from lane_detection import lane
from piracer.operate_radian import set_steering_radians
from piracer.vehicles import PiRacerStandard


if __name__ == '__main__':
  vehicle = PiRacerStandard()
  cap = cv2.VideoCapture(0)
  ret, original_frame = cap.read()
  while(cap.isOpened()):
    ret, original_frame = cap.read()
    if ret == False:
      break
    frame, radian = lane.lane_tracker(original_frame)
    cv2.imshow('frame', frame)
    set_steering_radians(piracer=vehicle ,radians=radian)
    if cv2.waitKey(10) == 27:
      break
  cap.release()
  cv2.destroyAllWindows()
