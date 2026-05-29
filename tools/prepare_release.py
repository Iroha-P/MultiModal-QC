import argparse
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT / "release"
CHUNK_SIZE = 512 * 1024 * 1024


SOURCE_DIRS = [
    "agents",
    "assets",
    "baselines",
    "data/scripts",
    "docs",
    "serve",
    "tests",
    "train",
    "tools",
]

SOURCE_FILES = [
    ".gitignore",
    "README.md",
    "README.zh-CN.md",
    "LICENSE",
    "config.py",
    "data/__init__.py",
    "docker-compose.yml",
    "requirements.txt",
    "oneclick.bat",
    "install.bat",
    "start.bat",
    "run_demo.bat",
    "stop.bat",
]

METADATA_FILES = [
    "data/mvtec_qa.json",
    "data/train.json",
    "data/val.json",
    "data/test.json",
]

BASELINE_PATHS = [
    "outputs/baselines/resnet",
    "runs/classify/outputs/baselines/yolo_defect/weights",
    "yolo26n.pt",
    "yolov8n-cls.pt",
]

PUBLIC_DOC_FILES = {
    "DEPLOYMENT.md",
    "Demo演示脚本.md",
    "FILE_GUIDE.md",
    "作品集展示页.md",
    "技术报告.md",
}


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    normalized = path.as_posix()
    if "superpowers" in parts:
        return True
    if "docs" in parts and path.suffix.lower() == ".md" and path.name not in PUBLIC_DOC_FILES:
        return True
    if normalized.startswith("data/drilling/") and path.name.endswith(".json"):
        return True
    if normalized == "data/drilling_qa.json":
        return True
    return any(
        marker in parts
        for marker in {
            ".git",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            ".specstory",
            "release",
            "storage",
        }
    )


def add_path(zf: zipfile.ZipFile, path: Path, arcname: Path | None = None):
    if not path.exists() or should_skip(path):
        return
    if arcname is None:
        arcname = path.relative_to(ROOT)
    if path.is_file():
        zf.write(path, arcname.as_posix())
        return
    for file in path.rglob("*"):
        if file.is_file() and not should_skip(file):
            zf.write(file, file.relative_to(ROOT).as_posix())


def zip_paths(zip_name: str, paths: list[str]):
    out = RELEASE_DIR / zip_name
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for item in paths:
            add_path(zf, ROOT / item)
    return out


def build_oneclick():
    paths = SOURCE_DIRS + SOURCE_FILES + METADATA_FILES
    return zip_paths("MultiModal-QC-oneclick.zip", paths)


def build_metadata():
    return zip_paths("MultiModal-QC-dataset-metadata.zip", METADATA_FILES)


def build_mvtec_sample():
    out = RELEASE_DIR / "MultiModal-QC-dataset-mvtec-sample.zip"
    sample_files = set(METADATA_FILES + ["data/mvtec/license.txt", "data/mvtec/readme.txt"])
    test_json = ROOT / "data" / "test.json"
    if test_json.exists():
        records = json.loads(test_json.read_text(encoding="utf-8"))
        for item in records:
            image_path = item.get("image")
            if image_path:
                sample_files.add(image_path.replace("\\", "/"))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for item in sorted(sample_files):
            add_path(zf, ROOT / item)
    return out


def build_lora():
    return zip_paths("MultiModal-QC-lora-best.zip", ["outputs/lora_defect/best"])


def build_baselines():
    return zip_paths("MultiModal-QC-baselines.zip", BASELINE_PATHS)


def split_file(src: Path, prefix: str, chunk_size: int = CHUNK_SIZE):
    if not src.exists():
        return []
    outputs = []
    with src.open("rb") as f:
        index = 1
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            out = RELEASE_DIR / f"{prefix}.part{index:03d}"
            with out.open("wb") as w:
                w.write(chunk)
            outputs.append(out)
            index += 1
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Build GitHub Release assets for MultiModal-QC.")
    parser.add_argument("--include-mvtec", action="store_true", help="Split data/mvtec/mvtec_anomaly_detection.tar.xz into release parts.")
    args = parser.parse_args()

    RELEASE_DIR.mkdir(exist_ok=True)

    built = [
        build_oneclick(),
        build_metadata(),
        build_mvtec_sample(),
        build_lora(),
        build_baselines(),
    ]

    if args.include_mvtec:
        built.extend(split_file(ROOT / "data" / "mvtec" / "mvtec_anomaly_detection.tar.xz", "mvtec_anomaly_detection.tar.xz"))

    print("Built release assets:")
    for item in built:
        size_mb = item.stat().st_size / 1024 / 1024
        print(f"- {item} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
