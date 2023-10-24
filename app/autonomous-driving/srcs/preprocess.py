# ------------------------------------------------------------------------------
# Library Import
import  cv2
import  numpy as np
import  tensorflow as tf
from    tensorflow.keras.utils \
        import  to_categorical

# Custom Library import
from    srcs.variables \
        import  *

# ------------------------------------------------------------------------------
def detect_orange_lines(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for orange color
    lower_orange = np.array([5, 50, 50])
    upper_orange = np.array([15, 255, 255])
    
    # Get mask of pixels within the defined range
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

# ------------------------------------------------------------------------------
# Preprocessor
def preprocessing(image):
    # Filtering : Convert to gray-scale and blur
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (3,3), 0)
    # Image resize
    image = image[HEIGHT_CUT:, :]
    image = cv2.resize(image, (WIDTH, HEIGHT))
    # Normalization
    image = image / 255.0
    return  image

# ------------------------------------------------------------------------------
# Image Loader
def load_image(csv, isTest=False):
    images = []
    labels = []

    for _, row in csv.iterrows():
        idx = row['index']
        str = row['steering']
        dir = row['direction(front-0/left-1/right-2)']
        pth = f"{FRAMES}/frame_{idx}_{str}.jpg"

        image = cv2.imread(pth, cv2.IMREAD_COLOR)
        image = preprocessing(pth)
        images.append(image)
        labels.append(dir)

    # One-hot encoding
    labels = to_categorical(labels, num_classes=3)

    return  np.array(images), np.array(labels)