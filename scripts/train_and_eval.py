from ultralytics import YOLO
from pathlib import Path
import json

# === Parameters — FOR YOURSELF (YOUR CASE) ===
DATA = "ds_custom.yaml"
BASE_MODEL = "yolov8n.pt"   
EPOCHS = 100
IMGSZ = 640
BATCH = 8
SEED = 42
CONF = 0.35                   # confidence threshold for validation/predictions
IOU = 0.5
OUT_DIR = Path("runs/custom/gear24_exp1")
PREDICT_TEST = True           # savings pictures with boxes test

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # === TRAIN ===
    model = YOLO(BASE_MODEL)
    train_res = model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        seed=SEED,
        project=str(OUT_DIR),
        name="train",
    )
    weights_path = Path(train_res.save_dir) / "weights" / "best.pt"

    # Cleane state, the best model reloading
    model = YOLO(str(weights_path))

    # === VALIDATE (VAL) ===
    print("\n=== VAL METRICS ===")
    metrics_val = model.val(
        data=DATA, split="val",
        conf=CONF, iou=IOU,
        save_json=True, plots=True,
        project=str(OUT_DIR), name="val_metrics"
    )
    print(metrics_val.results_dict)

    # === VALIDATE (TEST) ===
    print("\n=== TEST METRICS ===")
    metrics_test = model.val(
        data=DATA, split="test",
        conf=CONF, iou=IOU,
        save_json=True, plots=True,
        project=str(OUT_DIR), name="test_metrics"
    )
    print(metrics_test.results_dict)

    # === JSON ===
    summary = {
        "weights": str(weights_path),
        "conf": CONF, "iou": IOU,
        "imgsz": IMGSZ, "batch": BATCH, "seed": SEED,
        "val": metrics_val.results_dict,
        "test": metrics_test.results_dict
    }
    (OUT_DIR / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")

    # === VISUAL PREDICTION TEST ===
    if PREDICT_TEST:
        model.predict(
            source="dataset/images/test",
            conf=CONF, iou=IOU, imgsz=IMGSZ,
            save=True, save_txt=True, save_conf=True,
            project=str(OUT_DIR), name="pred_test", exist_ok=True
        )

    print(f"\nEverything is ready! Reports and files are here: {OUT_DIR}")

if __name__ == "__main__":
    main()
