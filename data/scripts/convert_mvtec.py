import json
import numpy as np
from pathlib import Path
from PIL import Image

DEFECT_NAMES = {
    "bottle": {"broken_large": "大面积破损", "broken_small": "小面积破损", "contamination": "污染"},
    "cable": {"bent_wire": "弯曲", "cable_swap": "线缆错位", "combined": "组合缺陷",
              "cut_inner_insulation": "内层绝缘切割", "cut_outer_insulation": "外层绝缘切割",
              "missing_cable": "缺失线缆", "missing_wire": "缺失线芯", "poke_insulation": "绝缘刺穿"},
    "capsule": {"crack": "裂纹", "faulty_imprint": "印记不良", "poke": "穿孔", "scratch": "划痕", "squeeze": "挤压"},
    "hazelnut": {"crack": "裂纹", "cut": "切口", "hole": "孔洞", "print": "印记"},
    "metal_nut": {"bent": "弯曲", "color": "颜色异常", "flip": "翻转", "scratch": "划痕"},
    "pill": {"color": "颜色异常", "combined": "组合缺陷", "contamination": "污染",
             "crack": "裂纹", "faulty_imprint": "印记不良", "scratch": "划痕", "type": "类型异常"},
    "screw": {"manipulated_front": "正面篡改", "scratch_head": "头部划痕", "scratch_neck": "颈部划痕",
              "thread_side": "螺纹侧面", "thread_top": "螺纹顶部"},
    "tile": {"crack": "裂纹", "glue_strip": "胶带", "gray_stroke": "灰色笔触", "oil": "油渍", "rough": "粗糙"},
    "toothbrush": {"defective": "缺陷"},
    "transistor": {"bent_lead": "弯曲引脚", "cut_lead": "切断引脚", "damaged_case": "外壳损坏", "misplaced": "错位"},
    "wood": {"color": "颜色异常", "combined": "组合缺陷", "hole": "孔洞", "liquid": "液体痕迹", "scratch": "划痕"},
    "zipper": {"broken_teeth": "断齿", "combined": "组合缺陷", "fabric_border": "面料边界",
               "fabric_interior": "面料内部", "rough": "粗糙", "split_teeth": "齿分离", "squeezed_teeth": "齿挤压"},
    "carpet": {"color": "颜色异常", "cut": "切口", "hole": "孔洞", "metal_contamination": "金属污染", "thread": "线头"},
    "grid": {"bent": "弯曲", "broken": "断裂", "glue": "胶水", "metal_contamination": "金属污染", "thread": "线头"},
    "leather": {"color": "颜色异常", "cut": "切口", "fold": "折痕", "glue": "胶水", "poke": "穿孔"},
}

def mask_to_location(mask: np.ndarray, h: int, w: int) -> str:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return "未知区域"
    cy, cx = ys.mean() / h, xs.mean() / w
    v = "上方" if cy < 0.33 else ("中部" if cy < 0.66 else "下方")
    ho = "左侧" if cx < 0.33 else ("中央" if cx < 0.66 else "右侧")
    return f"{ho}{v}"

def build_qa(category: str, image_path: str, is_good: bool, mask_path: str | None) -> dict:
    user_msg = "<image>\n请检查这个产品图片是否存在缺陷，如有请描述缺陷类型、位置和严重程度。"
    if is_good:
        assistant_msg = f"该{category}产品表面完好，无缺陷，判定合格。"
    else:
        location = "未知区域"
        defect_name = "表面缺陷"
        if mask_path and Path(mask_path).exists():
            mask = np.array(Image.open(mask_path).convert("L"))
            location = mask_to_location(mask, mask.shape[0], mask.shape[1])
            area_ratio = (mask > 0).sum() / mask.size
            severity = "高" if area_ratio > 0.05 else ("中等" if area_ratio > 0.01 else "轻微")
        else:
            severity = "中等"
        defect_folder = Path(image_path).parent.name if image_path else ""
        cat_defects = DEFECT_NAMES.get(category, {})
        defect_name = cat_defects.get(defect_folder, "表面缺陷")
        assistant_msg = f"检测到{defect_name}缺陷，位于产品{location}，严重程度：{severity}。建议进行质量复检。"
    return {
        "image": image_path,
        "conversations": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
    }

def convert_mvtec(mvtec_root: str, output_path: str):
    root = Path(mvtec_root)
    samples = []
    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        test_dir = category_dir / "test"
        gt_dir = category_dir / "ground_truth"
        if not test_dir.exists():
            continue
        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir():
                continue
            is_good = defect_dir.name == "good"
            for img in sorted(defect_dir.glob("*.png")):
                mask_path = None
                if not is_good and gt_dir.exists():
                    mask_candidates = list((gt_dir / defect_dir.name).glob(f"{img.stem}*mask*"))
                    if mask_candidates:
                        mask_path = str(mask_candidates[0])
                qa = build_qa(category, str(img), is_good, mask_path)
                samples.append(qa)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"Converted {len(samples)} samples -> {output_path}")
    return samples

if __name__ == "__main__":
    import sys
    mvtec_root = sys.argv[1] if len(sys.argv) > 1 else "data/mvtec"
    convert_mvtec(mvtec_root, "data/mvtec_qa.json")
