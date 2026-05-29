from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DB_PATH = PROJECT_ROOT / "multimodal_qc.db"

MVTEC_DIR = DATA_DIR / "mvtec"
DRILLING_DIR = DATA_DIR / "drilling"

BASE_MODEL = str(MODELS_DIR / "Qwen2-VL-2B-Instruct")
DEPLOY_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

LORA_DEFECT_PATH = OUTPUTS_DIR / "lora_defect"
LORA_BEST_PATH = OUTPUTS_DIR / "lora_defect" / "best"
LORA_DRILLING_PATH = OUTPUTS_DIR / "lora_drilling"

SCENE_TYPES = ["defect_3d", "drilling_compliance"]

for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, MVTEC_DIR, DRILLING_DIR]:
    d.mkdir(parents=True, exist_ok=True)
