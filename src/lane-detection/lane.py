import sys
import cv2
import math
import os
import numpy as np
import edge_detection as edge
import matplotlib.pyplot as plt

class Lane:
  def __init__(self, orig_frame):
    self.orig_frame = orig_frame
    # hold and image with the lane lines
    self.lane_line_markings = None
    # hold the image after perspective transformation
    self.warped_frame = None
    self.transformation_matrix = None
    self.inv_transformation_matrix = None
    self.orig_image_size = self.orig_frame.shape[::-1][1:]
    width = self.orig_image_size[0]
    height = self.orig_image_size[1]
    self.width = width
    self.height = height

    # road.1.mp4
    self.roi_points = np.float32([
      (420, 306), # Top-left corner
      (10, 537),   # Bottom-left corner
      (899, 537), # Bottom-right corner
      (599, 306)  # Top-right corner
    ])

    # road1.jpg
    # self.roi_points = np.float32([
    #   (476, 432), # Top-left corner
    #   (134, 672),   # Bottom-left corner
    #   (1066, 672), # Bottom-right corner
    #   (752, 432)  # Top-right corner
    # ])


    self.padding = int(0.25 * width)
    self.desired_roi_points = np.float32([
      [self.padding, 0], # Top-left corner
      [self.padding, self.orig_image_size[1]], # Bottom-left corner
      [self.orig_image_size[0] - self.padding, self.orig_image_size[1]], # Bottom-right corner
      [self.orig_image_size[0] - self.padding, 0] # Top-right corner
    ])

    # Histogram that shows the white pixels peaks for lane line detection
    self.histogram = None

    # sliding window parameters
    self.no_of_windows = 10
    self.margin = int((1/12) * width) # Window width is +/- margin
    self.minpix = int((1/24) * width) # Min no. of pixels to recenter window

    # Best fit polynomial lines for left line and right line of the lane
    self.left_fit = None
    self.right_fit = None
    self.left_lane_inds = None
    self.right_lane_inds = None
    self.ploty = None
    self.left_fitx = None
    self.right_fitx = None
    self.leftx = None
    self.rightx = None
    self.lefty = None
    self.righty = None

    # Pixel parameters for x and y dimensions
    self.YM_PER_PIX = 10.0 / 1000
    self.XM_PER_PIX = 3.7 / 781

    # Radii of curvature and offset
    self.left_curvem = None
    self.right_curvem = None
    self.center_offset = None

  """
    Isolates lane lines
    :return: Binary (i.e. black and white) image containing the lane lines
  """
  def get_line_markings(self, frame=None):
    if frame is None:
      frame = self.orig_frame

    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    # Perform Sobel edge detectation on the L (lightness)
    _, sxbinary = edge.threshold(hls[:, :, 1], thresh=(120, 255))
    sxbinary = edge.blur_gaussian(sxbinary, ksize=3)
    sxbinary = edge.mag_thresh(sxbinary, sobel_kernel=3, thresh=(110, 255))

    # Perform binary thresholding on the S (saturation)
    s_channel = hls[:, :, 2]
    _, s_binary = edge.threshold(s_channel, (80, 255))

    # Perform binary thresholding on the R (red)
    _, r_thresh = edge.threshold(frame[:, :, 0])

    rs_binary = cv2.bitwise_and(s_binary, r_thresh)

    self.lane_line_markings = cv2.bitwise_or(rs_binary, sxbinary.astype(np.uint8))

    return self.lane_line_markings

  """
    Plot the region of interest on an image

    :param frame: The current image frame
    :param plot: Plot the roi image if True
  """
  def plot_roi(self, frame=None, plot=False):
    if plot == False:
      return
    if frame is None:
      frame = self.orig_frame.copy()

    # Overlay trapezoid on the frame
    this_image = cv2.polylines(frame, np.int32([self.roi_points]), True, (140, 20, 255), 3)
    while(True):
      cv2.imshow('ROI image', this_image)
      if cv2.waitKey(0):
        break
    cv2.destroyAllWindows()

  """
    Perform the perspective transform

    :param frame: the current frame
    :param plot: Plot the roi image if True
    :return: Bird's eye view of the current lane
  """
  def perspective_transform(self, frame=None, plot=False):
    if frame is None:
      frame = self.lane_line_markings

    # Calculate the transform matrix
    self.transformation_matrix = cv2.getPerspectiveTransform(self.roi_points, self.desired_roi_points)

    # Calculate the inverse transformation matrix
    self.inv_transformation_matrix = cv2.getPerspectiveTransform(self.desired_roi_points, self.roi_points)

    # Perform the transform using the transformation matrix
    self.warped_frame = cv2.warpPerspective(frame, self.transformation_matrix, self.orig_image_size, flags=(cv2.INTER_LINEAR))

    # Convert image to binary
    (thresh, binary_warped) = cv2.threshold(self.warped_frame, 127, 255, cv2.THRESH_BINARY)
    self.warped_frame = binary_warped

    if plot == True:
      warped_copy = self.warped_frame.copy()
      warped_plot = cv2.polylines(warped_copy, np.int32([self.desired_roi_points]), True, (147, 20, 255), 3)
      while (True):
        cv2.imshow('Warped Image', warped_plot)
        if cv2.waitKey(0):
          break
      cv2.destroyAllWindows()

  """
    Calculate the image histogram to find peaks in white pixels count

    :param frame: the current image
    :param plot: create a plot if True
  """
  def calculate_histogram(self, frame=None, plot=False):
    if frame is None:
      frame = self.warped_frame

    # generate the histogram
    self.histogram = np.sum(frame[:frame.shape[0]:, :], axis=0)

    if plot == True:

      # Draw both the image and the histogram
      figure, (ax1, ax2) = plt.subplots(2,1) # 2 row, 1 columns
      figure.set_size_inches(10, 5)
      ax1.imshow(frame, cmap='gray')
      ax1.set_title("Warped Binary Frame")
      ax2.plot(self.histogram)
      ax2.set_title("Histogram Peaks")
      plt.show()

    return self.histogram

    """
    Get the left and right peak of the histogram

    Return the x coordinate of the left histogram peak and the right histogram
    peak.
    """
  def histogram_peak(self):
    midpoint = np.int32(self.histogram.shape[0]/2)
    leftx_base = np.argmax(self.histogram[:midpoint])
    rightx_base = np.argmax(self.histogram[midpoint:]) + midpoint

    # (x coordinate of left peak, x coordinate of right peak)
    return leftx_base, rightx_base


  """
    Get the indices of the lane line pixels using the sliding windows technique

    :param plot: show plot or not
    return Best fit lines for the left and right lines of the current lane
  """
  def get_lane_line_indices_sliding_windows(self, plot=True):
    # sliding window width is +/- margin
    margin = self.margin

    frame_sliding_window = self.warped_frame.copy()

    # Set the height of the sliding windows
    window_height = np.int32(self.warped_frame.shape[0] / self.no_of_windows)

    # Find the x and y coordinates of all the nonzero
    nonzero = self.warped_frame.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Store the pixel indices for the left and right lane lines
    left_lane_inds = []
    right_lane_inds = []

    # Current positions for pixel indices for each window
    leftx_base, rightx_base = self.histogram_peak()
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Go through one window at a time
    no_of_windows = self.no_of_windows

    for window in range(no_of_windows):
      # Identify window boundaries in x and y (and right and left)
      win_y_low = self.warped_frame.shape[0] - (window + 1) * window_height
      win_y_high = self.warped_frame.shape[0] - window * window_height

      win_xleft_low = leftx_current - margin
      win_xleft_high = leftx_current + margin
      win_xright_low = rightx_current - margin
      win_xright_high = rightx_current + margin
      # print(win_xleft_low, win_y_low, win_y_high)
      cv2.rectangle(frame_sliding_window, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (255, 255, 255), 2)
      cv2.rectangle(frame_sliding_window, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255, 255, 255), 2)

      # Identify the nonzero pixels in x and y within the window
      good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
      good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

      left_lane_inds.append(good_left_inds)
      right_lane_inds.append(good_right_inds)

      # If you found > minpix pixels, recenter next window on mean position
      minpix = self.minpix
      if len(good_left_inds) > minpix:
        leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
      if len(good_right_inds) > minpix:
        rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract the pixel coordinates for the left and right lane lines
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial curve to the pixel coordinates for the left and right
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    self.left_fit = left_fit
    self.right_fit = right_fit

    if plot == True:
      # Crate the x and y values to plot on the image
      ploty = np.linspace(0, frame_sliding_window.shape[0] - 1, frame_sliding_window.shape[0])
      left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
      right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

      # Generate an image to visualize the result
      out_img = np.dstack((frame_sliding_window, frame_sliding_window, (frame_sliding_window))) * 255

      # Add color to the left line pixels and right line pixels
      out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
      out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 0, 0]

      # Plot the figure with the sliding windows
      figure, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns
      ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
      ax2.imshow(frame_sliding_window, cmap='gray')
      ax3.imshow(out_img)
      ax3.plot(left_fitx, ploty, color='yellow')
      ax3.plot(right_fitx, ploty, color='yellow')
      ax1.set_title("Original Frame")
      ax2.set_title("Warped Frame with Sliding Windows")
      ax3.set_title("Detected Lane Lines with Sliding Windows")
      plt.tight_layout()
      plt.show()
    return self.left_fit, self.right_fit

  """
    Use the lane line from the previous sliding window to get the parameters
    for polynomial line for filling in the lane line

    :param left_fit: Polynomial function of the left lane line
    :param right_fit: Polynomial function of the right lane line
    :param plot: To display or not
  """
  def get_lane_line_previous_window(self, left_fit, right_fit, plot=False):
    margin = self.margin

    # Find the x and y coordinates of all the nonzero
    nonzero = self.warped_frame.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Store left and right lane pixel indices (quadratic equation)
    left_lane_inds = (
      (nonzerox > (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
      (nonzerox < (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] + margin))
      )
    right_lane_inds = (
      (nonzerox > (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
      (nonzerox < (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] + margin))
    )
    self.left_lane_inds = left_lane_inds
    self.right_lane_inds = right_lane_inds

    # Get the left and right lane line pixel locations (if True, value is pushed to array)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    self.leftx = leftx
    self.lefty = lefty
    self.rightx = rightx
    self.righty = righty

    # Fit a second order polynomial curve to each lane line
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    self.left_fit = left_fit
    self.right_fit = right_fit

    # Create the x and y values to plot on the image
    ploty = np.linspace(0, self.warped_frame.shape[0] - 1, self.warped_frame.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    self.ploty = ploty
    self.left_fitx = left_fitx
    self.right_fitx = right_fitx

    if plot == True:
      # Generate images to draw on
      out_img = np.dstack((self.warped_frame, self.warped_frame, (
                           self.warped_frame)))*255
      window_img = np.zeros_like(out_img)

      # Add color to the left and right line pixels
      out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
      out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [
                                                                     0, 0, 255]
      # Create a polygon to show the search window area, and recast
      # the x and y points into a usable format for cv2.fillPoly()
      margin = self.margin
      left_line_window1 = np.array([np.transpose(
        np.vstack([left_fitx-margin, ploty])
        )])
      left_line_window2 = np.array([np.flipud(
        np.transpose(np.vstack([left_fitx+margin, ploty]))
        )])
      left_line_pts = np.hstack((left_line_window1, left_line_window2))

      right_line_window1 = np.array([np.transpose(
        np.vstack([right_fitx-margin, ploty])
        )])
      right_line_window2 = np.array([np.flipud(
        np.transpose(np.vstack([right_fitx+margin, ploty]))
        )])
      right_line_pts = np.hstack((right_line_window1, right_line_window2))

      # Draw the lane onto the warped blank image
      cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
      cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
      result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

      # Plot the figures
      figure, (ax1, ax2, ax3) = plt.subplots(3,1) # 3 rows, 1 column
      figure.set_size_inches(10, 10)
      figure.tight_layout(pad=3.0)
      ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
      ax2.imshow(self.warped_frame, cmap='gray')
      ax3.imshow(result)
      ax3.plot(left_fitx, ploty, color='yellow')
      ax3.plot(right_fitx, ploty, color='yellow')
      ax1.set_title("Original Frame")
      ax2.set_title("Warped Frame")
      ax3.set_title("Warped Frame With Search Window")
      plt.show()


  """
    Overlay lane lines on the original frame
    :return: Lane with overlay
  """
  def overlay_lane_lines(self, plot=False):
    # Generate an image to draw the lane lines on
    warp_zero = np.zeros_like(self.warped_frame).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(
      np.vstack([self.left_fitx, self.ploty])
      )])

    pts_right = np.array([np.flipud(np.transpose(
      np.vstack([self.right_fitx, self.ploty])
      ))])

    pts = np.hstack((pts_left, pts_right))

    np.set_printoptions(threshold = sys.maxsize)
    # Draw lane on the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective
    newwarp = cv2.warpPerspective(color_warp, self.inv_transformation_matrix, (
      self.orig_frame.shape[1], self.orig_frame.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(self.orig_frame, 1, newwarp, 0.3, 0)

    if plot == True:
      # Plot the figures
      figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
      ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
      ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
      ax1.set_title("Original Frame")
      ax2.set_title("Original Frame With Lane Overlay")
      plt.tight_layout()
      plt.show()

    return result, newwarp

  """
    Calculate the road curvature in meters

    :param print_to_terminal Display data to console if True
    :return: Radii of curvature
  """
  def calculate_curvature(self, point_to_terminal=False):

    y_eval = np.max(self.ploty)

    left_fit_cr = np.polyfit(self.lefty * self.YM_PER_PIX, self.leftx * (self.XM_PER_PIX), 2)
    right_fit_cr = np.polyfit(self.righty * self.YM_PER_PIX, self.rightx * (self.XM_PER_PIX), 2)

    left_curvem = ((1 + (2*left_fit_cr[0] * y_eval * self.YM_PER_PIX + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curvem = ((1 + (2*right_fit_cr[0] * y_eval * self.YM_PER_PIX + right_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])

    if point_to_terminal == True:
      print(left_curvem, 'm', right_curvem, 'm')

    self.left_curvem = left_curvem
    self.right_curvem = right_curvem
    return left_curvem, right_curvem

  """
    Calculate the position of the car relative to the center

    :param print_to_terminal: Display data to console or not
    :return: Offset from the center of the lane
  """
  def calculate_car_position(self, print_to_terminal=False):

    # Get position of car in centimeters
    car_location = self.orig_frame.shape[1] / 2

    # Find the x coordinate of the lane line bottom
    height = self.orig_frame.shape[0]
    bottom_left = self.left_fit[0] * height**2 + self.left_fit[1]*height + self.left_fit[2]
    bottom_right = self.right_fit[0] * height**2 + self.right_fit[1]*height + self.right_fit[2]

    center_lane = (bottom_right - bottom_left) / 2 + bottom_left
    center_offset = (np.abs(car_location) - np.abs(center_lane)) * self.XM_PER_PIX * 100

    if print_to_terminal == True:
      print(str(center_offset) + 'cm')
    self.center_offset = center_offset
    return center_offset

  """
    Display curvature and offset statistics on the image

    :param: plot Display the plot if True
    :return: Image with lane lines and curvature
  """
  def display_curvature_offset(self, frame=None, plot=False):
    image_copy = None
    if frame is None:
      image_copy = self.orig_frame.copy()
    else:
      image_copy = frame

    cv2.putText(image_copy,'Curve Radius: '+str((
      self.left_curvem+self.right_curvem)/2)[:7]+' m', (int((
      5/600)*self.width), int((
      20/338)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, (float((
      0.5/600)*self.width)),(
      255,255,255),2,cv2.LINE_AA)

    cv2.putText(image_copy,'Center Offset: '+str(
      self.center_offset)[:7]+' cm', (int((
      5/600)*self.width), int((
      40/338)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, (float((
      0.5/600)*self.width)),(
      255,255,255),2,cv2.LINE_AA)

    if plot == True:
      cv2.imshow("Image with Curvature and Offset", image_copy)

    return image_copy

  def compute_steering_angle(self):
    x_offset = self.calculate_car_position()
    y_offset = int(self.height/ 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)
    print(angle_to_mid_radian)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg + 90

    return steering_angle


  """
    Draw the center line on the image
  """
  def display_heading_line(self, frame=None, steering_angle=None, plot=False):
    if frame is None:
      return

    heading_image = np.zeros_like(frame)

    steering_angle_radian = steering_angle / 180.0 * math.pi

    x1 = int(self.width / 2)
    y1 = self.height
    x2 = int(x1 - self.height / 2 / math.tan(steering_angle_radian))
    y2 = int(self.height / 1.5)

    cv2.line(heading_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    if plot == True:
      cv2.imshow("Heading Line", heading_image)
      cv2.waitKey(0)
      cv2.destroyAllWindows()
    return heading_image



def lane_tracker_video():
  cap = cv2.VideoCapture('./assets/road1.mp4')
  ret, original_frame = cap.read()
  while(cap.isOpened()):
    ret, original_frame = cap.read()
    if ret == False:
      break
    lane_obj = Lane(orig_frame=original_frame)

    # Perform thresholding to isolate lane lines
    lane_line_markings = lane_obj.get_line_markings()

    # Plot the region of interest on the iamge
    lane_obj.plot_roi(plot=False)

    # Perform the perspective transform to generate a bird's eye view
    warped_frame = lane_obj.perspective_transform(plot=False)

    # Generate the image histogram to serve as a starting point for finding lane line pixels
    histogram = lane_obj.calculate_histogram(plot=False)

    # Find lane line pixels using the sliding window method
    left_fit, right_fit = lane_obj.get_lane_line_indices_sliding_windows(plot=False)

    # Fill in the lane line
    lane_obj.get_lane_line_previous_window(left_fit, right_fit, plot=False)

    # Overlay lines on the original frame
    frame_with_lane_lines, save_img = lane_obj.overlay_lane_lines(plot=False)

    steering_angle = lane_obj.compute_steering_angle()

    ret = lane_obj.display_heading_line(frame=frame_with_lane_lines, steering_angle=steering_angle, plot=False)

    cv2.imshow('frame', ret)
    if cv2.waitKey(10) == 27:
      break
  cap.release()
  cv2.destroyAllWindows()

def lane_tracker_image():
  original_frame = cv2.imread('./assets/road1.jpg')
  lane_obj = Lane(orig_frame=original_frame)

  # Perform thresholding to isolate lane lines
  lane_line_markings = lane_obj.get_line_markings()

  # Plot the region of interest on the iamge
  lane_obj.plot_roi(plot=False)

  # Perform the perspective transform to generate a bird's eye view
  warped_frame = lane_obj.perspective_transform(plot=False)

  # Generate the image histogram to serve as a starting point for finding lane line pixels
  histogram = lane_obj.calculate_histogram(plot=False)

  # Find lane line pixels using the sliding window method
  left_fit, right_fit = lane_obj.get_lane_line_indices_sliding_windows(plot=False)

  # Fill in the lane line
  lane_obj.get_lane_line_previous_window(left_fit, right_fit, plot=False)

  # Overlay lines on the original frame
  frame_with_lane_lines, save_img = lane_obj.overlay_lane_lines(plot=False)

  steering_angle = lane_obj.compute_steering_angle()

  lane_obj.display_heading_line(frame=frame_with_lane_lines, steering_angle=steering_angle, plot=True)

if __name__ == '__main__':
  lane_tracker_video()
