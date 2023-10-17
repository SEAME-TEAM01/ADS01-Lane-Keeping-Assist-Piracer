import cv2
from lane_detection.lane import Lane
# import piracer


if __name__ == '__main__':
  # vehicle = piracer.vehicles.PiRacerStandard()
  lane = Lane();


  i = 1
  while(i < 282):
    original_frame = cv2.imread('images/%d.jpg' % i)
    degree, radian, frame = lane.lane_tracker(original_frame)
    cv2.imwrite("dataset/frame02_%04d_%04d.jpg" % (i, degree), original_frame)
    print(i, radian)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    i += 1

  cv2.destroyAllWindows()


# if __name__ == '__main__':
#   lane = Lane();

#   i = 0
#   original_frame = cv2.imread(f'dataset/frame_0107.jpg')
#   degree, radian, frame = lane.lane_tracker(original_frame)
#   print(i, radian)
#   cv2.imshow('frame', frame)
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()
