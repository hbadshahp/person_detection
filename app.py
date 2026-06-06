from flask import Flask, Response,render_template, request
import cv2
import json
import time
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
import threading

latest_frame = None
latest_annotated = None

app = Flask(__name__)

model = YOLO("yolov8n.pt")

def load_polygon():

    with open("static/polygon.json", "r") as f:
        data = json.load(f)

    return Polygon(data["polygon"])

polygon = load_polygon()

RTSP_URL = "rtsp://admin:cctv%40999@192.168.0.185:554/Streaming/Channels/102"

cap = cv2.VideoCapture(RTSP_URL)

def camera_reader():

    global latest_frame

    while True:

        ret, frame = cap.read()

        if ret:
            latest_frame = frame.copy()

def detector():

    global latest_frame
    global latest_annotated

    while True:

        if latest_frame is None:
            continue

        frame = latest_frame.copy()

        results = model(
            frame,
            classes=[0],
            imgsz=320,
            conf=0.4,
            verbose=False
        )

        annotated = frame.copy()

        # Put your polygon and NG logic here

        latest_annotated = annotated

threading.Thread(
    target=camera_reader,
    daemon=True
).start()

threading.Thread(
    target=detector,
    daemon=True
).start()

def generate_frames():

    while True:

        success, frame = cap.read()
        print(frame.shape)
        if not success or frame is None:
            continue

        start = time.time()
        results = model(frame, classes=[0], verbose=False)
        print("YOLO Time:", time.time() - start)
        annotated = frame.copy()

        pts = list(polygon.exterior.coords)

        for i in range(len(pts)-1):

            p1 = tuple(map(int, pts[i]))
            p2 = tuple(map(int, pts[i+1]))

            cv2.line(
                annotated,
                p1,
                p2,
                (0,0,255),
                3
            )

        for box in results[0].boxes:

            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()

            x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])

            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)

            inside = polygon.contains(Point(cx,cy))

            color = (0,255,0)

            if inside:
                color = (0,0,255)

            cv2.rectangle(
                annotated,
                (x1,y1),
                (x2,y2),
                color,
                2
            )

            cv2.circle(
                annotated,
                (cx,cy),
                5,
                color,
                -1
            )

            status = "NG" if inside else "OK"

            cv2.putText(
                annotated,
                status,
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

        ret, buffer = cv2.imencode('.jpg', annotated)

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )

@app.route("/polygon")
def polygon_page():
    return render_template("polygon.html")

@app.route("/snapshot")
def snapshot():

    ret, frame = cap.read()

    _, buffer = cv2.imencode('.jpg', frame)

    return Response(
        buffer.tobytes(),
        mimetype='image/jpeg'
    )


@app.route("/save_polygon", methods=["POST"])
def save_polygon():

    data = request.get_json()

    with open("static/polygon.json", "w") as f:
        json.dump(data, f, indent=4)

    return "Polygon Saved Successfully"

@app.route('/')
def index():
    return """
    <html>
    <body>
        <h2>Hikvision Live Stream</h2>
        <img src="/video_feed" width="1280">
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)
