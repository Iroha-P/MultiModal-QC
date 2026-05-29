import json
import re
from pathlib import Path
from agents.schema import InspectionReport

PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def parse_perception(text: str) -> dict:
    """Parse the VLM's natural language output into structured data."""
    result = {
        "has_defect": False,
        "verdict": "合格",
        "defect_type": "",
        "location": "",
        "severity": "",
        "details": text.strip(),
        "confidence": 0.9,
    }

    if "合格" in text and "不合格" not in text and "缺陷" not in text.split("合格")[0]:
        result["has_defect"] = False
        result["verdict"] = "合格"
        return result

    defect_types = {
        "裂纹": "crack", "划痕": "scratch", "凹坑": "dent", "污染": "contamination",
        "变形": "deformation", "破损": "broken", "弯曲": "bent", "篡改": "manipulated",
        "缺损": "missing", "胶水": "glue", "切口": "cut", "孔洞": "hole",
        "折痕": "fold", "颜色异常": "color", "缺陷": "defect",
    }
    for cn, en in defect_types.items():
        if cn in text:
            result["has_defect"] = True
            result["verdict"] = "不合格"
            result["defect_type"] = cn
            break

    loc_match = re.search(r"位于[^\s，。、]*", text)
    if loc_match:
        result["location"] = loc_match.group()

    for level in ["高", "中等", "轻微"]:
        if f"严重程度：{level}" in text or f"严重程度: {level}" in text:
            result["severity"] = level
            break

    if result["has_defect"]:
        result["confidence"] = 0.85
    return result


class InspectionPipeline:
    def __init__(self, model, scene_type: str):
        self.model = model
        self.scene_type = scene_type
        prefix = "defect" if scene_type == "defect_3d" else "drilling"
        self.perceive_prompt = load_prompt(f"{prefix}_perceive")

    def run(self, input_path: str) -> InspectionReport:
        ext = Path(input_path).suffix.lower()
        is_video = ext in (".mp4", ".avi", ".mov", ".mkv")
        img = None if is_video else input_path
        vid = input_path if is_video else None

        # Stage 1: Perception (VLM inference)
        perception_raw = self.model.chat(
            self.perceive_prompt, image_path=img, video_path=vid, max_new_tokens=256
        )

        # Stage 2-4: Rule-based parsing (no extra model calls)
        parsed = parse_perception(perception_raw)

        defects = []
        if parsed["has_defect"]:
            defects.append({
                "type": parsed["defect_type"],
                "location": parsed["location"],
                "severity": parsed["severity"],
            })

        detection_raw = json.dumps({
            "has_defect": parsed["has_defect"],
            "defect_type": parsed["defect_type"],
            "location": parsed["location"],
            "confidence": parsed["confidence"],
        }, ensure_ascii=False)

        decision_raw = json.dumps({
            "verdict": parsed["verdict"],
            "reasons": [parsed["details"]],
        }, ensure_ascii=False)

        suggestion = "建议进行质量复检。" if parsed["has_defect"] else "产品合格，可放行。"

        return InspectionReport(
            scene_type=self.scene_type,
            verdict=parsed["verdict"],
            confidence=parsed["confidence"],
            defects=defects,
            suggestion=suggestion,
            perception_raw=perception_raw,
            detection_raw=detection_raw,
            decision_raw=decision_raw,
        )
