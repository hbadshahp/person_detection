import cv2

url = "rtsp://admin:cctv%40999@192.168.0.185:554/cam/realmonitor?channel=1&subtype=0"

cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to receive frame")
        break

    cv2.imshow("Hikvision", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
