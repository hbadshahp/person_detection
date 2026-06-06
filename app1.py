from flask import Flask, Response, render_template, request
import cv2
import json
import threading
import time
import torch

print("CUDA:", torch.cuda.is_available())
print("GPU :", torch.cuda.get_device_name(0))

from ultralytics import YOLO
from shapely.geometry import Point, Polygon

app = Flask(__name__)

# -------------------------------
# CONFIG
# -------------------------------

RTSP_URL = "rtsp://admin:cctv%40999@192.168.0.185:554/Streaming/Channels/102"

# -------------------------------
# LOAD MODEL
# -------------------------------

model = YOLO("yolov8n.pt")
model.to("cuda")
# -------------------------------
# LOAD POLYGON
# -------------------------------

def load_polygon():
    try:
        with open("static/polygon.json", "r") as f:
            data = json.load(f)

        return Polygon(data["polygon"])

    except Exception as e:
        print("Polygon load error:", e)

        return Polygon([
            (100, 100),
            (500, 100),
            (500, 500),
            (100, 500)
        ])

polygon = load_polygon()

# -------------------------------
# CAMERA
# -------------------------------

cap = cv2.VideoCapture(RTSP_URL)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

latest_frame = None
latest_annotated = None

# -------------------------------
# CAMERA THREAD
# -------------------------------

def camera_reader():

    global latest_frame

    while True:

        ret, frame = cap.read()

        if ret:
            latest_frame = frame.copy()

            if latest_frame is not None:
                print("Camera OK")

        else:
            time.sleep(0.05)

# -------------------------------
# YOLO THREAD
# -------------------------------

def detector():

    global latest_frame
    global latest_annotated
    global polygon

    while True:

        if latest_frame is None:
            print("Detector OK")
            time.sleep(0.01)
            continue

        frame = latest_frame.copy()

        start = time.time()

        results = model(
            frame,
            classes=[0],     # person only
            imgsz=320,
            conf=0.4,
            device=0,
            verbose=False
        )

        yolo_time = time.time() - start

        annotated = frame.copy()

        # Draw polygon
        pts = list(polygon.exterior.coords)

        for i in range(len(pts) - 1):

            p1 = tuple(map(int, pts[i]))
            p2 = tuple(map(int, pts[i + 1]))

            cv2.line(
                annotated,
                p1,
                p2,
                (0, 0, 255),
                3
            )

        # Person detection
        for box in results[0].boxes:

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            inside = polygon.contains(
                Point(cx, cy)
            )

            color = (0, 255, 0)
            status = "OK"

            if inside:
                color = (0, 0, 255)
                status = "NG"

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.circle(
                annotated,
                (cx, cy),
                5,
                color,
                -1
            )

            cv2.putText(
                annotated,
                status,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        cv2.putText(
            annotated,
            f"YOLO: {yolo_time:.2f}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        if annotated is not None:
            latest_annotated = annotated.copy()
            print("Detector OK")

        time.sleep(0.01)

# -------------------------------
# START THREADS
# -------------------------------

threading.Thread(
    target=camera_reader,
    daemon=True
).start()

threading.Thread(
    target=detector,
    daemon=True
).start()

# -------------------------------
# FLASK VIDEO STREAM
# -------------------------------

def generate_frames():

    global latest_annotated

    while True:

        if latest_annotated is None:
            time.sleep(0.1)
            continue

        try:

            frame = latest_annotated.copy()

            if frame is None:
                continue

            ret, buffer = cv2.imencode(
                '.jpg',
                frame
            )

            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                frame_bytes +
                b'\r\n'
            )

        except Exception as e:

            print("STREAM ERROR:", e)

            time.sleep(0.1)

# -------------------------------
# ROUTES
# -------------------------------

@app.route('/')
def index():

    return """
    <html>
    <body>
        <h2>Person Intrusion Detection</h2>
        <img src="/video_feed" width="1280">
        <br><br>
        <a href="/polygon">Polygon Setup</a>
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/polygon')
def polygon_page():

    return render_template(
        'polygon.html'
    )

@app.route('/snapshot')
def snapshot():

    global latest_frame

    if latest_frame is None:
        return "No Frame"

    _, buffer = cv2.imencode(
        '.jpg',
        latest_frame
    )

    return Response(
        buffer.tobytes(),
        mimetype='image/jpeg'
    )

@app.route(
    '/save_polygon',
    methods=['POST']
)
def save_polygon():

    global polygon

    data = request.get_json()

    with open(
        "static/polygon.json",
        "w"
    ) as f:

        json.dump(
            data,
            f,
            indent=4
        )

    polygon = Polygon(
        data["polygon"]
    )

    return "Polygon Saved Successfully"

# -------------------------------
# MAIN
# -------------------------------

if __name__ == "__main__":

    app.run(
        host='0.0.0.0',
        port=5000,
        threaded=True,
        debug=False
    )
