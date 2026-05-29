import json
import tempfile
from pathlib import Path

def test_mask_to_location():
    from data.scripts.convert_mvtec import mask_to_location
    import numpy as np
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[10:50, 180:240] = 255
    loc = mask_to_location(mask, 256, 256)
    assert "右" in loc
    assert "上" in loc

def test_build_qa_good():
    from data.scripts.convert_mvtec import build_qa
    qa = build_qa("bottle", "test.png", is_good=True, mask_path=None)
    assert qa["image"] == "test.png"
    assert "合格" in qa["conversations"][1]["content"]

def test_build_qa_defect():
    from data.scripts.convert_mvtec import build_qa
    import numpy as np
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[100:200, 100:200] = 255
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        from PIL import Image
        Image.fromarray(mask).save(f.name)
        qa = build_qa("bottle", "test.png", is_good=False, mask_path=f.name)
    assert "缺陷" in qa["conversations"][1]["content"]
