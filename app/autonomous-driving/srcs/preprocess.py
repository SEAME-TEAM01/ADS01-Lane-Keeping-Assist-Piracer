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
# Preprocessor
def preprocessing(pth, isTest=False):
    # Image Load
    image = cv2.imread(pth, cv2.IMREAD_COLOR)
    # Filtering : Convert to gray-scale and blur
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # Image resize
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

        image = preprocessing(pth, isTest=isTest)
        images.append(image)
        labels.append(dir)

    # One-hot encoding
    labels = to_categorical(labels, num_classes=3)

    return  np.array(images), np.array(labels)