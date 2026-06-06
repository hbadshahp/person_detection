import cv2
import json

points=[]

url="rtsp://admin:cctv%40999@192.168.0.183:554/cam/realmonitor?channel=1&subtype=0"

cap=cv2.VideoCapture(url)

ret,frame=cap.read()

if not ret:
    print("Camera failed")
    exit()

def mouse(event,x,y,flags,param):

    global points

    if event==cv2.EVENT_LBUTTONDOWN:

        points.append((x,y))

        print(points)

while True:

    img=frame.copy()

    for p in points:
        cv2.circle(img,p,5,(0,0,255),-1)

    if len(points)>1:
        cv2.polylines(
            img,
            [__import__('numpy').array(points)],
            False,
            (0,255,0),
            2
        )

    cv2.imshow("Select ROI",img)

    cv2.setMouseCallback("Select ROI",mouse)

    key=cv2.waitKey(1)

    if key==ord('s'):

        with open("roi.json","w") as f:
            json.dump(points,f)

        print("Saved")
        break

cv2.destroyAllWindows()
