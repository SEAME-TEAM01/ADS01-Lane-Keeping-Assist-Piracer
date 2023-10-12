import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

class Lane:
  def __init__(self):
    self.curr_steering_angle = 90
    self.original_frame = None

  # Drawing function
  def display_heading_line(self, frame, steering_angle, line_color=(0, 0, 255), line_width=5 ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
    return heading_image

  def display_lines(self, frame, lines, line_color=(0, 255, 0), line_width=4, plot=False):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    if plot == True:
      cv2.imshow('Lane Detection', line_image)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
    return line_image

  def draw_line_segments(self, original_frame, line_segments):
    line_color = (0, 255, 0)
    line_thickness = 2

    if line_segments is not None:
        for line_segment in line_segments:
            x1, y1, x2, y2 = line_segment[0]
            cv2.line(original_frame, (x1, y1), (x2, y2), line_color, line_thickness)

    plt.imshow(original_frame)
    plt.show()

  # Thresholding function
  def Canny(self, frame, thresh=(128, 255)):
    return cv2.Canny(frame, thresh[0], thresh[1]);

  def blur_gaussian(self, frame, ksize=3):
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)

    # Steering angle calculation
  def calc_steering_angle(self, frame, lane_lines):
    if len(lane_lines) == 0:
        return -90

    height, width, _ = frame.shape
    if len(lane_lines) == 1:
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
    else:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        mid = int(width / 2 * 1)
        x_offset = (left_x2 + right_x2) / 2 - mid

    y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
    steering_angle = angle_to_mid_deg + 90  # this is the steering angle needed by piracer front wheel
    return steering_angle

  def stabilize_steering_angle(
          self,
          curr_steering_angle,
          new_steering_angle,
          num_of_lane_lines,
          max_angle_deviation_two_lines=5,
          max_angle_deviation_one_lane=1):

    if num_of_lane_lines == 2 :
        max_angle_deviation = max_angle_deviation_two_lines
    else :
        max_angle_deviation = max_angle_deviation_one_lane

    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle
            + max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_steering_angle = new_steering_angle
    return stabilized_steering_angle

  # Lane detection function
  def detect_line_segments(self, cropped_edges, plot=False):
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold,
                                    np.array([]), minLineLength=90, maxLineGap=30)

    if plot == True:
      self.draw_line_segments(self.original_frame, line_segments)
    return line_segments

  def make_points(self, frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2.5)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

  def average_slope_intercept(self, frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/2
    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary
    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0.3:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))
    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        lane_lines.append(self.make_points(frame, left_fit_average))
    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        lane_lines.append(self.make_points(frame, right_fit_average))

    return lane_lines

  def region_of_interest(self, edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (0, height * 1 / 4),
        (width, height * 1 / 4),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges

  def lane_tracker(self, original_frame):
    self.original_frame = original_frame
    gaussian_frame = self.blur_gaussian(original_frame)

    hsv = cv2.cvtColor(gaussian_frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([0, 40, 40])
    upper_orange = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    frame = self.Canny(mask, thresh=(100, 200))

    cropped_edges = self.region_of_interest(frame)

    line_segments = self.detect_line_segments(cropped_edges, plot=False)

    lane_lines = self.average_slope_intercept(original_frame, line_segments)
    lane_lines_image = self.display_lines(original_frame, lane_lines, plot=False)

    steering_angle = self.calc_steering_angle(original_frame, lane_lines)
    heading_image = self.display_heading_line(lane_lines_image, steering_angle)
    self.curr_steering_angle = self.stabilize_steering_angle(self.curr_steering_angle, steering_angle, len(lane_lines))

    return self.curr_steering_angle * math.pi / 180, heading_image
