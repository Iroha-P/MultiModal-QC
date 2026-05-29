from agents.schema import Perception, Detection, Decision, InspectionReport

def test_perception_serialization():
    p = Perception(product_type="bottle", appearance="cylindrical, transparent",
        anomalies=[{"type": "scratch", "location": "top-right", "size": "small"}])
    d = p.to_dict()
    assert d["product_type"] == "bottle"
    assert len(d["anomalies"]) == 1

def test_inspection_report():
    report = InspectionReport(scene_type="defect_3d", verdict="不合格", confidence=0.92,
        defects=[{"type": "scratch", "location": "top-right", "severity": "high"}],
        suggestion="建议返工", perception_raw="...", detection_raw="...", decision_raw="...")
    d = report.to_dict()
    assert d["verdict"] == "不合格"
    assert d["confidence"] == 0.92
