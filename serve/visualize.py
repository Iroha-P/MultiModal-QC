"""Attention visualization for all models: GradCAM heatmaps + text-based annotation."""
import re
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms


def gradcam_resnet(model_dict: dict, img_path: str, output_path: str) -> str:
    model = model_dict["model"]
    device = model_dict["device"]
    transform = model_dict["transform"]

    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])

    img_pil = Image.open(img_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    grayscale_cam = cam(input_tensor=img_tensor)[0]

    img_resized = np.array(img_pil.resize((224, 224))).astype(np.float32) / 255.0
    vis = show_cam_on_image(img_resized, grayscale_cam, use_rgb=True)

    orig = np.array(img_pil)
    h, w = orig.shape[:2]
    vis_full = cv2.resize(vis, (w, h))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(vis_full, cv2.COLOR_RGB2BGR))
    return output_path


def gradcam_yolo(yolo_model, img_path: str, output_path: str) -> str:
    """Generate activation heatmap for YOLOv8 classification model."""
    import torch.nn.functional as F

    torch_model = yolo_model.model.model
    img_pil = Image.open(img_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    device = next(torch_model.parameters()).device
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    activations = {}
    def hook_fn(module, input, output):
        activations['feat'] = output

    handle = torch_model[8].register_forward_hook(hook_fn)

    with torch.no_grad():
        x = img_tensor
        for layer in torch_model:
            x = layer(x)
            if isinstance(x, tuple):
                x = x[0]

    handle.remove()

    feat = activations['feat']
    if isinstance(feat, tuple):
        feat = feat[0]
    heatmap = feat.squeeze(0).mean(dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    img_resized = np.array(img_pil.resize((224, 224))).astype(np.float32) / 255.0
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    vis = show_cam_on_image(img_resized, heatmap_resized, use_rgb=True)

    orig = np.array(img_pil)
    h, w = orig.shape[:2]
    vis_full = cv2.resize(vis, (w, h))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(vis_full, cv2.COLOR_RGB2BGR))
    return output_path


POSITION_MAP = {
    ("左", "上"): (0.0, 0.0, 0.4, 0.4),
    ("左", "中"): (0.0, 0.3, 0.4, 0.7),
    ("左", "下"): (0.0, 0.6, 0.4, 1.0),
    ("中", "上"): (0.3, 0.0, 0.7, 0.4),
    ("中", "中"): (0.25, 0.25, 0.75, 0.75),
    ("中", "下"): (0.3, 0.6, 0.7, 1.0),
    ("右", "上"): (0.6, 0.0, 1.0, 0.4),
    ("右", "中"): (0.6, 0.3, 1.0, 0.7),
    ("右", "下"): (0.6, 0.6, 1.0, 1.0),
    ("中央", "上"): (0.25, 0.0, 0.75, 0.45),
    ("中央", "下"): (0.25, 0.55, 0.75, 1.0),
    ("中央", "中"): (0.2, 0.2, 0.8, 0.8),
}


def parse_position(text: str):
    h_pos = "中"
    v_pos = "中"
    if "左侧" in text or "左" in text:
        h_pos = "左"
    elif "右侧" in text or "右" in text:
        h_pos = "右"
    elif "中央" in text:
        h_pos = "中央"

    if "上方" in text or "上" in text:
        v_pos = "上"
    elif "下方" in text or "下" in text:
        v_pos = "下"
    elif "中部" in text or "中间" in text:
        v_pos = "中"

    return POSITION_MAP.get((h_pos, v_pos), (0.2, 0.2, 0.8, 0.8))


def annotate_qwen(img_path: str, text: str, output_path: str) -> str:
    img = cv2.imread(img_path)
    if img is None:
        return img_path
    h, w = img.shape[:2]

    if "合格" in text and "不合格" not in text and "缺陷" not in text.split("合格")[0]:
        color = (0, 200, 0)
        cv2.putText(img, "OK", (w // 2 - 40, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
        cv2.rectangle(img, (10, 10), (w - 10, h - 10), color, 3)
    else:
        rx1, ry1, rx2, ry2 = parse_position(text)
        x1, y1 = int(w * rx1), int(h * ry1)
        x2, y2 = int(w * rx2), int(h * ry2)

        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        img = cv2.addWeighted(overlay, 0.2, img, 0.8, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

        defect_type = ""
        for dt in ["裂纹", "划痕", "凹坑", "污染", "变形", "破损", "弯曲", "篡改", "胶水", "切口", "缺损", "折痕"]:
            if dt in text:
                defect_type = dt
                break

        severity = ""
        for s in ["高", "中等", "轻微"]:
            if s in text:
                severity = s
                break

        label = f"Defect: {defect_type}"
        if severity:
            label += f" ({severity})"

        font_scale = max(0.5, min(w, h) / 600)
        thickness = max(1, int(font_scale * 2))

        # Use PIL for Chinese text rendering
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        cn_label = ""
        if defect_type:
            cn_label = f"缺陷: {defect_type}"
        if severity:
            cn_label += f" | 严重程度: {severity}"
        if not cn_label:
            cn_label = "检测到缺陷"

        try:
            font = ImageFont.truetype("msyh.ttc", max(16, int(min(w, h) / 25)))
        except Exception:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", max(16, int(min(w, h) / 25)))
            except Exception:
                font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), cn_label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - th - 12, x1 + tw + 12, y1], fill=(220, 50, 50))
        draw.text((x1 + 6, y1 - th - 8), cn_label, fill=(255, 255, 255), font=font)
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, img)
    return output_path
