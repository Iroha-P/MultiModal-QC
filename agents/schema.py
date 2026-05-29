from dataclasses import dataclass, field, asdict

@dataclass
class Perception:
    product_type: str = ""
    appearance: str = ""
    anomalies: list[dict] = field(default_factory=list)
    def to_dict(self):
        return asdict(self)

@dataclass
class Detection:
    has_defect: bool = False
    defect_type: str = ""
    location: str = ""
    confidence: float = 0.0
    details: str = ""
    def to_dict(self):
        return asdict(self)

@dataclass
class Decision:
    verdict: str = "合格"
    reasons: list[str] = field(default_factory=list)
    suggestion: str = ""
    def to_dict(self):
        return asdict(self)

@dataclass
class InspectionReport:
    scene_type: str = ""
    verdict: str = ""
    confidence: float = 0.0
    defects: list[dict] = field(default_factory=list)
    suggestion: str = ""
    perception_raw: str = ""
    detection_raw: str = ""
    decision_raw: str = ""
    def to_dict(self):
        return asdict(self)
