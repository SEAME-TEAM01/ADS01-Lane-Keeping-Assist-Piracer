import  cv2

frame = cv2.VideoCapture(0)

while frame.isOpened():
    rst, img = frame.read()
    if img is None:
        print("Image Load Failure")
        break
    print(img)
    cv2.imgshow('freme', img)
    act = cv2.waitKey(10) & 0xFF

    # press q to exit
    if act == ord('q') or act == 113:
        break

frame.release()
cv2.destroyAllWindows()

