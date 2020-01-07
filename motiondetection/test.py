import cv2

cap = cv2.VideoCapture(0)

while(True):
    t, f = cap.read()
    if t:
        cv2.imshow("hi", f)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
