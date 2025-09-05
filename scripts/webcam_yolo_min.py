import time
from pathlib import Path
import cv2
from ultralytics import YOLO

# ==== Settings ====
WEIGHTS = "PUT YOUR PATH HERE"  # Path to weights
CAM_INDEX = 0                                                 # Camera index (0/1/...)
CONF = 0.35
IOU = 0.5
IMG_SIZE = 640
CLASS_NAMES = ["gear_24t", "RB_2x6", "WP_1x8", "YSB_2x2"]  # Class names
USE_DSHOW = True                               # True: it can be faster for Windows OS

# ==== Colors/Fonts ====
COLOR_BOX = (40, 220, 40)
COLOR_TEXT = (255, 255, 255)

# ==== Class for indicators control ====
class PlacementIndicator:
    def __init__(self):
        self.indicators = {
            "RB_2x6": {
                "position": "top_right",
                "active_color": (0, 0, 255),  # Red
                "inactive_color": (0, 0, 0),   # Black
                "is_active": False
            },
            "gear_24t": {
                "position": "top_left",
                "active_color": (0, 255, 0),   # Green
                "inactive_color": (0, 0, 0),
                "is_active": False
            },
            "WP_1x8": {
                "position": "bottom_left",
                "active_color": (255, 0, 0),   # Blue
                "inactive_color": (0, 0, 0),
                "is_active": False
            },
            "YSB_2x2": {
                "position": "bottom_right",
                "active_color": (255, 255, 0), # Yellow
                "inactive_color": (0, 0, 0),
                "is_active": False
            }
        }
    
    def update_detection(self, class_name, detected):
        """Status detection update"""
        if class_name in self.indicators:
            self.indicators[class_name]["is_active"] = detected
    
    def reset_all(self):
        """Clean all indicators"""
        for indicator in self.indicators.values():
            indicator["is_active"] = False
    
    def draw_indicators(self, frame):
        """Write indecators in the frame (camera)"""
        height, width = frame.shape[:2]
        
        # Dimensions Big Rectangular
        container_width = 300
        container_height = 200
        margin = 20
        
        # Position for Big Rectangular
        container_x = width - container_width - margin
        container_y = height - container_height - margin
        
        # Big grey rectangular
        cv2.rectangle(frame, 
                     (container_x, container_y),
                     (container_x + container_width, container_y + container_height),
                     (100, 100, 100), -1)  # Серый цвет
        
        # Frame drawing around main container/box
        cv2.rectangle(frame,
                     (container_x, container_y),
                     (container_x + container_width, container_y + container_height),
                     (200, 200, 200), 2)  # Светло-серая рамка
        
        # Dimensions for small rectangulars
        small_width = 120
        small_height = 80
        padding = 20
        
        # Positions for 4 rectangulars
        positions = {
            "top_left": (container_x + padding, container_y + padding),
            "top_right": (container_x + container_width - padding - small_width, container_y + padding),
            "bottom_left": (container_x + padding, container_y + container_height - padding - small_height),
            "bottom_right": (container_x + container_width - padding - small_width, 
                           container_y + container_height - padding - small_height)
        }
        
        # Draw small rectangulars and underwrites
        for class_name, indicator in self.indicators.items():
            pos_name = indicator["position"]
            x, y = positions[pos_name]
            
            # Color choose, depends on activity
            color = indicator["active_color"] if indicator["is_active"] else indicator["inactive_color"]
            
            # Rectangular drawing
            cv2.rectangle(frame,
                         (x, y),
                         (x + small_width, y + small_height),
                         color, -1)
            
            # Frame drawing
            cv2.rectangle(frame,
                         (x, y),
                         (x + small_width, y + small_height),
                         (255, 255, 255), 2)
            
            # Text with detail name 
            text = class_name
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            text_x = x + (small_width - text_width) // 2
            text_y = y + small_height // 2 + text_height // 2
            
            cv2.putText(frame, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        return height, width

def open_camera(index=0, use_dshow=False):
    if use_dshow and hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not to open camera {index}")
    return cap

def main():
    model = YOLO(WEIGHTS)
    cap = open_camera(CAM_INDEX, USE_DSHOW)
    indicator = PlacementIndicator()

    t0 = time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Clean indicators before new frame
        indicator.reset_all()
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Inference (quite mode without logs)
        res = model.predict(frame, conf=CONF, iou=IOU, imgsz=IMG_SIZE, verbose=False)[0]

        detected_classes = set()
        
        # Draw boxes and indicator updating
        if res.boxes is not None and len(res.boxes) > 0:
            for xyxy, cls_id, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                cid = int(cls_id) if cls_id is not None else 0
                score = float(conf)
                
                # Get name of class
                class_name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)
                detected_classes.add(class_name)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)

                label = class_name + f" {score:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), COLOR_BOX, -1)
                cv2.putText(frame, label, (x1 + 3, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1, cv2.LINE_AA)
        
        # inticator uptade for detected classes 
        for class_name in detected_classes:
            indicator.update_detection(class_name, True)
        
        # Drawing indicators
        indicator.draw_indicators(frame)

        # down headline
        cv2.putText(frame, "Detection System - Place parts in highlighted areas", 
                   (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # FPS evry 10 frames
        frames += 1
        if frames % 10 == 0:
            t1 = time.time()
            fps = 10.0 / max(1e-6, (t1 - t0))
            t0 = t1
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("YOLO Detection System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()