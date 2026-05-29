import json
import os
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from config import DB_PATH, SCENE_TYPES, BASE_MODEL, LORA_BEST_PATH
from serve.database import Database

db = Database(str(DB_PATH))
models_cache = {}
pipeline_cache = {}
load_errors = {}

RESNET_PATH = "outputs/baselines/resnet/resnet50_defect.pt"
YOLO_PATH = "runs/classify/outputs/baselines/yolo_defect/weights/best.pt"


def load_resnet():
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(RESNET_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return {"model": model, "transform": transform, "device": device}


def load_yolo():
    from ultralytics import YOLO
    return YOLO(YOLO_PATH)


def predict_resnet(img_path: str) -> dict:
    import torch
    from PIL import Image
    r = models_cache["resnet"]
    img = Image.open(img_path).convert("RGB")
    tensor = r["transform"](img).unsqueeze(0).to(r["device"])
    with torch.no_grad():
        out = r["model"](tensor)
        probs = torch.softmax(out, dim=1)[0]
    pred = probs.argmax().item()
    conf = probs[pred].item()
    verdict = "合格" if pred == 0 else "不合格"
    return {"verdict": verdict, "confidence": round(conf, 4), "model": "ResNet50"}


def predict_yolo(img_path: str) -> dict:
    model = models_cache["yolo"]
    results = model(img_path, verbose=False)
    top1 = results[0].probs.top1
    conf = results[0].probs.top1conf.item()
    name = results[0].names[top1]
    verdict = "合格" if name == "good" else "不合格"
    return {"verdict": verdict, "confidence": round(conf, 4), "model": "YOLOv8"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    if os.getenv("MULTIMODAL_QC_SKIP_QWEN") == "1":
        load_errors["qwen"] = "Skipped by MULTIMODAL_QC_SKIP_QWEN=1."
    elif not (Path(BASE_MODEL) / "config.json").exists():
        load_errors["qwen"] = f"Base model not found: {BASE_MODEL}"
    elif not LORA_BEST_PATH.exists():
        load_errors["qwen"] = f"LoRA adapter not found: {LORA_BEST_PATH}"
    else:
        try:
            print("Loading Qwen2-VL + LoRA...", flush=True)
            from agents.model import QwenVLModel
            from agents.pipeline import InspectionPipeline
            qwen = QwenVLModel(BASE_MODEL, lora_path=str(LORA_BEST_PATH), use_4bit=True)
            models_cache["qwen"] = qwen
            for scene in SCENE_TYPES:
                pipeline_cache[scene] = InspectionPipeline(qwen, scene)
        except Exception as exc:
            load_errors["qwen"] = str(exc)

    if Path(RESNET_PATH).exists():
        try:
            print("Loading ResNet50...", flush=True)
            models_cache["resnet"] = load_resnet()
        except Exception as exc:
            load_errors["resnet"] = str(exc)
    else:
        load_errors["resnet"] = f"Weight not found: {RESNET_PATH}"

    if Path(YOLO_PATH).exists():
        try:
            print("Loading YOLOv8...", flush=True)
            models_cache["yolo"] = load_yolo()
        except Exception as exc:
            load_errors["yolo"] = str(exc)
    else:
        load_errors["yolo"] = f"Weight not found: {YOLO_PATH}"

    print(f"Loaded models: {list(models_cache.keys())}", flush=True)
    if load_errors:
        print(f"Model load warnings: {load_errors}", flush=True)
    yield

app = FastAPI(title="MultiModal-QC API", lifespan=lifespan)

VIS_DIR = Path("storage/vis")
VIS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/vis", StaticFiles(directory=str(VIS_DIR)), name="vis")


@app.get("/health")
def health():
    return {"status": "ok", "models": list(models_cache.keys()), "load_errors": load_errors}


@app.get("/scenes")
def list_scenes():
    return {"scenes": SCENE_TYPES}


@app.get("/inspections")
def list_inspections(limit: int = 50):
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM inspections ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return {"inspections": [dict(r) for r in rows]}


@app.get("/inspections/{insp_id}")
def get_inspection(insp_id: int):
    insp = db.get_inspection(insp_id)
    results = db.get_results(insp_id)
    defects = db.get_defects(insp_id)
    return {"inspection": insp, "results": results, "defects": defects}


@app.post("/quick_inspect")
async def quick_inspect(
    scene_type: str = Form(...),
    file: UploadFile = File(...),
    model_type: str = Form("qwen"),
):
    upload_dir = Path("storage/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    insp_id = db.create_inspection(scene_type, str(file_path))

    try:
        if model_type not in models_cache:
            detail = load_errors.get(model_type, "Model is not loaded.")
            db.update_status(insp_id, "failed")
            return JSONResponse(status_code=503, content={"error": detail, "model": model_type})

        from serve.visualize import gradcam_resnet, gradcam_yolo, annotate_qwen
        vis_name = f"{uuid.uuid4().hex}.jpg"
        vis_path = str(VIS_DIR / vis_name)

        if model_type == "resnet":
            result = predict_resnet(str(file_path))
            text = f"ResNet50 判定：{result['verdict']}（置信度 {result['confidence']:.2%}）"
            gradcam_resnet(models_cache["resnet"], str(file_path), vis_path)
        elif model_type == "yolo":
            result = predict_yolo(str(file_path))
            text = f"YOLOv8 判定：{result['verdict']}（置信度 {result['confidence']:.2%}）"
            gradcam_yolo(models_cache["yolo"], str(file_path), vis_path)
        else:
            qwen = models_cache["qwen"]
            prompt = "请检查这个产品图片是否存在缺陷，如有请描述缺陷类型、位置和严重程度。"
            text = qwen.chat(prompt, image_path=str(file_path), max_new_tokens=256)
            annotate_qwen(str(file_path), text, vis_path)
            result = {"verdict": "", "confidence": 0, "model": "Qwen2-VL QLoRA"}

        vis_url = f"/vis/{vis_name}"
        db.add_result(insp_id, model_type, text, float(result.get("confidence") or 0))
        db.update_status(insp_id, "completed")
        return {"inspection_id": insp_id, "result": text, "mode": "quick",
                "model": model_type, "vis_url": vis_url, **result}
    except Exception as e:
        db.update_status(insp_id, "failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/inspect")
async def create_inspection(scene_type: str = Form(...), file: UploadFile = File(...)):
    if scene_type not in SCENE_TYPES:
        return JSONResponse(status_code=400, content={"error": f"Invalid scene: {scene_type}"})
    upload_dir = Path("storage/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    insp_id = db.create_inspection(scene_type, str(file_path))
    db.update_status(insp_id, "running")
    if scene_type in pipeline_cache:
        pipe = pipeline_cache[scene_type]
        try:
            report = pipe.run(str(file_path))
            from serve.visualize import annotate_qwen
            vis_name = f"{uuid.uuid4().hex}.jpg"
            vis_path = str(VIS_DIR / vis_name)
            annotate_qwen(str(file_path), report.perception_raw, vis_path)
            vis_url = f"/vis/{vis_name}"

            db.add_result(insp_id, "perceive", report.perception_raw, report.confidence)
            db.add_result(insp_id, "detect", report.detection_raw, report.confidence)
            db.add_result(insp_id, "decide", report.decision_raw, report.confidence)
            db.add_result(insp_id, "report", json.dumps(report.to_dict(), ensure_ascii=False), report.confidence)
            for defect in report.defects:
                db.add_defect(insp_id, defect.get("type",""), defect.get("location",""), defect.get("severity",""), report.suggestion)
            db.update_status(insp_id, "completed")
            result = report.to_dict()
            result["vis_url"] = vis_url
            return {"inspection_id": insp_id, "report": result}
        except Exception as e:
            db.update_status(insp_id, "failed")
            return JSONResponse(status_code=500, content={"error": str(e)})
    else:
        db.update_status(insp_id, "pending")
        return {"inspection_id": insp_id, "status": "pending", "message": "Model not loaded."}
