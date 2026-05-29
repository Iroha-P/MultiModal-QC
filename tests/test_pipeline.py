from unittest.mock import MagicMock

from agents.pipeline import InspectionPipeline
from agents.schema import InspectionReport


def make_mock_model():
    model = MagicMock()
    model.chat = MagicMock(return_value="clear product image, no obvious anomaly")
    return model


def test_pipeline_defect():
    model = make_mock_model()
    pipe = InspectionPipeline(model, "defect_3d")
    report = pipe.run("test.png")

    assert isinstance(report, InspectionReport)
    assert report.scene_type == "defect_3d"
    assert report.confidence == 0.9
    assert report.defects == []
    assert model.chat.call_count == 1


def test_pipeline_drilling():
    model = make_mock_model()
    pipe = InspectionPipeline(model, "drilling_compliance")
    report = pipe.run("test.mp4")

    assert isinstance(report, InspectionReport)
    assert report.scene_type == "drilling_compliance"
    assert report.confidence == 0.9
    assert report.defects == []
    assert model.chat.call_count == 1
