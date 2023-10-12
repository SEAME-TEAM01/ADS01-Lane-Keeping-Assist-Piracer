import cv2
from lane_detection.lane import Lane
import piracer


if __name__ == '__main__':
  vehicle = piracer.vehicles.PiRacerStandard()
  lane = Lane();
  cap = cv2.VideoCapture(0)

  cap.set(cv2.CAP_PROP_FPS, 20)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)


  i = 0
  while(cap.isOpened()):
    ret, original_frame = cap.read()
    if ret == False:
      break
    original_frame = cv2.flip(original_frame, 0)
    radian, frame = lane.lane_tracker(original_frame)
    print(radian, end=' ')
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    piracer.operate_radian.set_steering_radians(piracer=vehicle ,radian=radian)

  cap.release()
  cv2.destroyAllWindows()
