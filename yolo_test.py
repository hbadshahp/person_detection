from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
polygon = load_polygon()

rtsp_url = "rtsp://admin:cctv%40999@192.168.0.185:554/Streaming/Channels/101"

cap = cv2.VideoCapture(rtsp_url)

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    results = model(frame, verbose=False)

    annotated = results[0].plot()

    display_frame = cv2.resize(annotated, (1024, 576))

    cv2.imshow("YOLO", display_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
