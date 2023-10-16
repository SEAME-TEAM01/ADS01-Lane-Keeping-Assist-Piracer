# - import python library
import cv2
import numpy as np

# - variables
VIDEO = 0

# - filter
VIDEO = 0

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def process_image(frame):
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color ranges
    lower_white = np.array([0,0,200])
    upper_white = np.array([180,255,255])
    lower_orange = np.array([5, 50, 50])
    upper_orange = np.array([15, 255, 255])
    
    # Create masks
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    mask = cv2.bitwise_or(mask_white, mask_orange)
    
    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Canny Edge Detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Define ROI
    height = frame.shape[0]
    polygons = np.array([
        [(0, height), (800, height), (400, 330)]
    ])
    roi = region_of_interest(edges, polygons)

    # Hough Line Transform
    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, threshold=50, minLineLength=10, maxLineGap=250)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return frame

# - main
if  __name__ == "__main__":
    capture = cv2.VideoCapture(VIDEO)

    while True:
        rst, frame = capture.read()
        if  not rst:
            break
        frame = cv2.flip(frame, -1)
    
        result = process_image(frame)
        
        # cv2.imshow('Camera Feed', result)
        cv2.imwrite('output.jpg', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()
