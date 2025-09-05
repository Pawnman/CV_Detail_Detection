import time
from pathlib import Path
import cv2
from ultralytics import YOLO

# ==== Settings ====
WEIGHTS = "PASTE YOUR WEIGHT"  # Path to weights
CAM_INDEX = 0                  # camera index (0/1/...)
CONF = 0.35
IOU = 0.5
IMG_SIZE = 640
CLASS_NAMES = ["gear_24t"]     
USE_DSHOW = True               # True: for Windows OS it can open camera faster

# ==== Colors/Fonts ====
COLOR_BOX = (40, 220, 40)
COLOR_TEXT = (255, 255, 255)

def open_camera(index=0, use_dshow=False):
    if use_dshow and hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"I could not open the camera {index}")
    return cap

def main():
    model = YOLO(WEIGHTS)
    cap = open_camera(CAM_INDEX, USE_DSHOW)

    t0 = time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Иinference (quite mode without logs)
        res = model.predict(frame, conf=CONF, iou=IOU, imgsz=IMG_SIZE, verbose=False)[0]

        # Boxes drawing
        if res.boxes is not None and len(res.boxes) > 0:
            for xyxy, cls_id, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                cid = int(cls_id) if cls_id is not None else 0
                score = float(conf)
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)

                label = (CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)) + f" {score:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), COLOR_BOX, -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1, cv2.LINE_AA)

        # FPS every 10 seconds
        frames += 1
        if frames % 10 == 0:
            t1 = time.time()
            fps = 10.0 / max(1e-6, (t1 - t0))
            t0 = t1
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("YOLO webcam (gear_24t)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
