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
    print(i, radian, end=' ')
    # frame_filename = f'dataset/frame_{i:04d}.jpg'
    # cv2.imwrite(frame_filename, original_frame)
    cv2.imshow('frame', frame)
    i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    piracer.operate_radian.set_steering_radians(piracer=vehicle ,radian=radian)

  cap.release()
  cv2.destroyAllWindows()

# if __name__ == '__main__':
#   # vehicle = piracer.vehicles.PiRacerStandard()
#   lane = Lane();
#   cap = cv2.VideoCapture('assets/piracer_pov04.mp4')

#   cap.set(cv2.CAP_PROP_FPS, 20)
#   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)


#   i = 0
#   while(i < 500):
#     ret, original_frame = cap.read()
#     if ret == False:
#       break
#     # original_frame = cv2.flip(original_frame, 0)
#     radian, frame = lane.lane_tracker(original_frame)
#     # frame_filename = f'dataset/frame_{i:04d}.jpg'
#     # cv2.imwrite(frame_filename, original_frame)
#     print(i, radian, end=' ')
#     cv2.imshow('frame', frame)
#     i += 1
#     cv2.waitKey(0)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#       break
#     piracer.operate_radian.set_steering_radians(piracer=None ,radian=radian)

#   cap.release()
#   cv2.destroyAllWindows()

# 55, 72, 73, 77, 80, 83, 87,273, 275, 276, 278, 282,

# if __name__ == '__main__':
#   lane = Lane();

#   i = 0
#   original_frame = cv2.imread(f'dataset/frame_0077.jpg')
#   radian, frame = lane.lane_tracker(original_frame)
#   print(i, radian)
#   cv2.imshow('frame', frame)
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()
