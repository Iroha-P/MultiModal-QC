import json
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def qa_to_yolo_dataset(qa_path: str, output_dir: str, img_size: int = 640):
    """Convert QA JSON to YOLO classification dataset structure.

    YOLO classification expects:
      output_dir/train/good/xxx.png
      output_dir/train/defect/xxx.png
      output_dir/val/good/xxx.png
      output_dir/val/defect/xxx.png
    """
    out = Path(output_dir)
    with open(qa_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    split = "train" if "train" in qa_path else "val"
    for cls in ["good", "defect"]:
        (out / split / cls).mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        img_path = sample["image"]
        label_text = sample["conversations"][1]["content"]
        is_good = "合格" in label_text and "不合格" not in label_text
        cls = "good" if is_good else "defect"
        dst = out / split / cls / f"{i:05d}.png"
        if not dst.exists():
            try:
                shutil.copy2(img_path, dst)
            except FileNotFoundError:
                continue

    print(f"Converted {len(samples)} samples to {out / split}")


def train_yolo(dataset_dir: str, output_dir: str, epochs: int = 20, img_size: int = 640):
    model = YOLO("yolov8n-cls.pt")
    results = model.train(
        data=dataset_dir,
        epochs=epochs,
        imgsz=img_size,
        batch=32,
        project=output_dir,
        name="yolo_defect",
        exist_ok=True,
    )
    return results


def evaluate_yolo(model_path: str, qa_test_path: str, output_path: str):
    model = YOLO(model_path)
    with open(qa_test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    preds = []
    labels = []
    for sample in test_data:
        img_path = sample["image"]
        label_text = sample["conversations"][1]["content"]
        is_good = "合格" in label_text and "不合格" not in label_text
        labels.append(0 if is_good else 1)

        try:
            results = model(img_path, verbose=False)
            top1_cls = results[0].probs.top1
            class_names = results[0].names
            pred_name = class_names[top1_cls]
            preds.append(0 if pred_name == "good" else 1)
        except Exception:
            preds.append(0)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"YOLOv8 results: {metrics}")
    return metrics


if __name__ == "__main__":
    qa_to_yolo_dataset("data/train.json", "data/yolo_cls")
    qa_to_yolo_dataset("data/val.json", "data/yolo_cls")
    train_yolo("data/yolo_cls", "outputs/baselines")
    best_model = "runs/classify/outputs/baselines/yolo_defect/weights/best.pt"
    evaluate_yolo(best_model, "data/test.json", "outputs/baselines/yolo/yolo_results.json")
