from flask import Flask, Response, render_template, request, redirect, url_for, session
import cv2
import json
import threading
import time
import torch

from datetime import datetime
last_alarm_time = 0

def save_alarm():

    global last_alarm_time

    if time.time() - last_alarm_time < 10:
        return

    last_alarm_time = time.time()

    try:

        with open(
            "static/alarms.json",
            "r"
        ) as f:

            alarms = json.load(f)

    except:

        alarms = []

    alarms.insert(0, {

    "time":
    datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    ),

    "event":
    "Person Intrusion",

    "status":
    "NG"
})

    alarms = alarms[:100]

    with open(
        "static/alarms.json",
        "w"
    ) as f:

        json.dump(
            alarms,
            f,
            indent=4
        )
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")

from ultralytics import YOLO
from shapely.geometry import Point, Polygon

app = Flask(__name__)
app.secret_key = "iota2026"

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "iota2026"

USER_USERNAME = "user"
USER_PASSWORD = "123456"

# -------------------------------
# CONFIG
# -------------------------------

RTSP_URL = "rtsp://admin:cctv%40999@192.168.0.185:554/Streaming/Channels/102"

# -------------------------------
# LOAD MODEL
# -------------------------------

model = YOLO("yolov8n.pt")

DEVICE = 0 if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    model.to("cuda")
    print("Running YOLO on GPU")
else:
    model.to("cpu")
    print("Running YOLO on CPU")
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
    global cap

    while True:

        ret, frame = cap.read()

        if ret:
            latest_frame = frame.copy()

        else:

            print("Camera disconnected. Reconnecting...")

            cap.release()

            time.sleep(2)

            cap = cv2.VideoCapture(RTSP_URL)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# -------------------------------
# YOLO THREAD
# -------------------------------

def detector():

    global latest_frame
    global latest_annotated
    global polygon

    while True:

        if latest_frame is None:
            time.sleep(0.01)
            continue

        frame = latest_frame.copy()

        start = time.time()

        results = model(
            frame,
            classes=[0],
            imgsz=320,
            conf=0.4,
            device=DEVICE,
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
                save_alarm()

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

@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':

        username = request.form.get('username')
        password = request.form.get('password')

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:

            session['logged_in'] = True
            session['role'] = 'admin'
            return redirect('/')

        elif username == USER_USERNAME and password == USER_PASSWORD:

            session['logged_in'] = True
            session['role'] = 'user'
            return redirect('/')

        return redirect('/login')

    return """
    <!DOCTYPE html>
    <html>
    <head>
    <title>Login</title>

    <style>

body{
    margin:0;
    padding:0;
    height:100vh;
    display:flex;
    justify-content:center;
    align-items:center;
    background:linear-gradient(135deg,#eef2f7,#dce6f2);
    font-family:Arial,sans-serif;
    }

.login-box{
    width:450px;
    background:white;
    padding:45px;
    border-radius:12px;
    box-shadow:0 0 20px rgba(0,0,0,0.15);
    text-align:center;
}

.logo{
    width:160px;
    height:auto;
    margin-bottom:20px;
}

h2{
    margin-bottom:25px;
    color:#333;
}

input{
    width:100%;
    padding:12px;
    margin-top:10px;
    margin-bottom:15px;
    border:1px solid #ccc;
    border-radius:6px;
    box-sizing:border-box;
}

button{
    width:100%;
    padding:12px;
    border:none;
    background:#007bff;
    color:white;
    font-size:16px;
    border-radius:6px;
    cursor:pointer;
}

button:hover{
    background:#0056b3;
}

</style>

</head>

<body>

<div class="login-box">

    <img src="/static/logo.png" class="logo">

    <h3 style="color:#666;margin-top:0;">
    RGAC INDIA Engineering & Automation
    </h3>

    <h2>Person Intrusion Detection</h2>

    <form method="POST">

        <input type="text"
               name="username"
               placeholder="Username"
               required>

        <input type="password"
               name="password"
               placeholder="Password"
               required>

        <button type="submit">
            Login
        </button>

    </form>

</div>

</body>
</html>
"""
@app.route('/logout')
def logout():

    session.clear()

    return redirect('/login')
    
@app.route('/')
def index():

    if not session.get('logged_in'):
        return redirect('/login')

    role = session.get('role')

    return render_template(
        "dashboard.html",
        role=role
    )

@app.route('/video_feed')
def video_feed():

    if not session.get('logged_in'):
        return redirect('/login')

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/polygon')
def polygon_page():

    if not session.get('logged_in'):
        return redirect('/login')

    if session.get('role') != 'admin':
        return "Admin Access Required", 403

    return render_template('polygon.html')
    

@app.route('/snapshot')
def snapshot():

    if not session.get('logged_in'):
        return redirect('/login')

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

@app.route('/save_polygon', methods=['POST'])
def save_polygon():

    if not session.get('logged_in'):
        return "Unauthorized", 401

    if session.get('role') != 'admin':
        return "Admin Access Required", 403

    global polygon

    data = request.get_json()

    with open("static/polygon.json", "w") as f:
        json.dump(data, f, indent=4)

    polygon = Polygon(data["polygon"])

    return "Polygon Saved Successfully"

@app.route('/alarms')
def alarms():

    if not session.get('logged_in'):
        return redirect('/login')

    try:

        with open(
            "static/alarms.json",
            "r"
        ) as f:

            data = json.load(f)

    except:

        data = []

    return render_template(
        "alarms.html",
        alarms=data
    )
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
