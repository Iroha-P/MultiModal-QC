import pytest
import tempfile
from pathlib import Path

def test_create_inspection():
    from serve.database import Database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = Database(f.name)
        db.init_db()
        insp_id = db.create_inspection("defect_3d", "test/image.png")
        assert insp_id == 1
        insp = db.get_inspection(insp_id)
        assert insp["scene_type"] == "defect_3d"
        assert insp["status"] == "pending"

def test_add_result():
    from serve.database import Database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = Database(f.name)
        db.init_db()
        insp_id = db.create_inspection("drilling_compliance", "test/video.mp4")
        db.add_result(insp_id, "perceive", '{"description": "test"}', 0.95)
        results = db.get_results(insp_id)
        assert len(results) == 1
        assert results[0]["agent_step"] == "perceive"
        assert results[0]["confidence"] == 0.95

def test_add_defect():
    from serve.database import Database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = Database(f.name)
        db.init_db()
        insp_id = db.create_inspection("defect_3d", "test/image.png")
        db.add_defect(insp_id, "scratch", "top-right", "high", "replace part")
        defects = db.get_defects(insp_id)
        assert len(defects) == 1
        assert defects[0]["defect_type"] == "scratch"

def test_update_status():
    from serve.database import Database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = Database(f.name)
        db.init_db()
        insp_id = db.create_inspection("defect_3d", "test/image.png")
        db.update_status(insp_id, "completed")
        insp = db.get_inspection(insp_id)
        assert insp["status"] == "completed"
        assert insp["completed_at"] is not None
