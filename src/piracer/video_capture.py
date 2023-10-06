import cv2


cap = cv2.VideoCapture(0)  


cap.set(cv2.CAP_PROP_FPS, 20)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  


fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))  

while True:
    ret, frame = cap.read()  

    if not ret:
        break

    
    frame = cv2.flip(frame, 0)  

    out.write(frame)  

    cv2.imshow('frame', frame)  

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()  
out.release()  
cv2.destroyAllWindows()  
