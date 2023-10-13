import cv2
import os
from piracer.cameras import Camera
from piracer.vehicles import PiRacerStandard
import numpy as np
import time

def nothing(x):
    pass

if __name__ == '__main__':
    os.system(f'v4l2-ctl -d /dev/video0 --set-fmt-video=width=640,height=400,pixelformat=MJPG --set-parm 10 --stream-count 1')
    piracer = PiRacerStandard()

    camera = Camera()
    width  = camera.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = camera.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    LR = [0, 640]
    prev_time = time.time()

    speed = 0.3
    steering = 0.0

    cv2.namedWindow("LANE")
    cv2.createTrackbar('low_H', 'LANE', 0, 179, nothing)
    cv2.createTrackbar('low_S', 'LANE', 0, 255, nothing)
    cv2.createTrackbar('low_V', 'LANE', 0, 255, nothing)
    cv2.createTrackbar('high_H', 'LANE', 0, 179, nothing)
    cv2.createTrackbar('high_S', 'LANE', 0, 255, nothing)
    cv2.createTrackbar('high_V', 'LANE', 0, 255, nothing)
    
    cv2.setTrackbarPos('low_H', 'LANE', 20)
    cv2.setTrackbarPos('low_S', 'LANE', 70)
    cv2.setTrackbarPos('low_V', 'LANE', 80)
    cv2.setTrackbarPos('high_H', 'LANE', 50)
    cv2.setTrackbarPos('high_S', 'LANE', 180)
    cv2.setTrackbarPos('high_V', 'LANE', 140)

    piracer.set_throttle_percent(speed)
    
    while 1:
        # Calculate FPS
        now_time = time.time()
        fps = 1/(now_time - prev_time)
        prev_time = time.time()
        # print(fps)
        image = camera.read_image()
        
        # HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        low_H = cv2.getTrackbarPos('low_H', 'LANE')
        low_S = cv2.getTrackbarPos('low_S', 'LANE')
        low_V = cv2.getTrackbarPos('low_V', 'LANE')
        high_H = cv2.getTrackbarPos('high_H', 'LANE')
        high_S = cv2.getTrackbarPos('high_S', 'LANE')
        high_V = cv2.getTrackbarPos('high_V', 'LANE')
        lower = np.array([low_H, low_S, low_V])
        upper = np.array([high_H, high_S, high_V])
        yellow_mask = cv2.inRange(hsv, lower, upper)
        
        masked = cv2.bitwise_and(image, image, mask = yellow_mask)
        # cv2.imshow('LANE', masked)
        
        
        # Canny
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        kernel = 7
        blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        canny = cv2.Canny(gray, 50, 150)
        # cv2.imshow('Canny', canny)

        # Hough Line
        linesP = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, None, 20, 10)
        L_lain = []
        R_lain = []
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                # cv2.line(image, (l[0], l[1]), (l[2], l[3]), (0,255,0), 3, cv2.LINE_AA)

                slope = (l[1]-l[3])/(l[0]-l[2]+0.01)

                if (slope<-0.5 or 0.5<slope):
                    pass
                else:
                    continue

                if (height*0.5<l[1]<height*0.8 and height*0.5<l[3]<height*0.8):
                    pass
                else:
                    continue
                
                mid = (l[0]+l[2])/2
                if mid<(width*0.4):
                    L_lain.append(mid)
                elif (width*0.6)<mid:
                    R_lain.append(mid)
                cv2.line(image, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

                if L_lain:
                    LR[0] = sum(L_lain)/len(L_lain)
                    # LR[0] = min(L_lain)

                if R_lain:
                    LR[1] = sum(R_lain)/len(R_lain)
                    # LR[1] = min(R_lain)

        midpoint= (LR[0]+LR[1])/2

        angle = (midpoint-width//2)/(width//2)*5
        angle = float(angle)
        if 1.0<angle:
            angle = 1.0
        elif angle<-1.0:
            angle = -1.0
        else:
            angle = round(angle,1)
        
        if ((steering*angle) < 0):
            steering = 0
        else:
            steering = angle
            
        # Move Car
        # piracer.set_throttle_percent(speed)
        piracer.set_steering_percent(-steering)
        print(f'L = {LR[0]}\tR = {LR[1]}')
        print(f'midpoint = {round(midpoint,2)}\tsteer = {angle}')
        print(f'fps = {fps}\n')
        # cv2.imshow('LANE', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break