import cv2 

cam = cv2.VideoCapture(1)

while True:
    _, frame = cam.read()

    cv2.imshow('Video', frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows()
        break