import cv2
import numpy as np

"""
Description: A collection of methods to detect help with edge detection

Return a 2D binary array(mask) in which all pixels are either 0 or 1

:param array: NumPy 2D array that we want to convert to binary values
:param thresh: values used for thresholding
:param value Output value when between the supplied threshold
:return: Binary 2D array
         number of row x number of columns
         number of pixels from top to bottom x number of pixels from left to right
"""
def binary_array(array, thresh, value=0):
  if value == 0:
    binary = np.ones_like(array)
  else:
    binary = np.zeros_like(array)
    value = 1
  binary[(array >= thresh[0]) & (array <= thresh[1])] = value
  return binary

"""
  Implementation for Gaussian blur to reduce noise and detail in the image

  :param image: 2D or 3D array to be blurred
  :param ksize: Size of the small matrix (i.e. kernel) used to blur
  :return: Blurred 2D image

"""
def blur_gaussian(image, ksize=3):
  return cv2.GaussianBlur(image, (ksize, ksize), 0)

"""
  Implementation of Sobel edge detection

  :param image: 2D or 3D array to be blurred
  :param sobel_kernel: Size of the small matrix
  :param thresh: values used for thresholding
  return: Binary (Black and white) 2D mask image
"""
def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
  sobelx = np.absolute(sobel(image, orient='x', sobel_kernel=sobel_kernel))
  sobely = np.absolute(sobel(image, orient='y', sobel_kernel=sobel_kernel))
  mag = np.sqrt(sobelx**2 + sobely**2)
  return binary_array(mag, thresh)

"""
  Find edges that are aligned vertically and horizontally on the image

  :param img_channel: Channel from an image
  :param orient: Across which axis of the image are we detecting edges
  :param sobel_kernel: Number of rows and columns of the kernel
  :return: Image with sobel edge detection applied
"""
def sobel(img_channel, orient='x', sobel_kernel=3):
  if orient == 'x':
    sobel = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, sobel_kernel)
  elif orient == 'y':
    sobel = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, sobel_kernel)
  return sobel

def threshold(channel, thresh=(128, 255), thresh_type=cv2.THRESH_BINARY):
  return cv2.threshold(channel, thresh[0], thresh[1], thresh_type)
