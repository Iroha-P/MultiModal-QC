import json
import random
from pathlib import Path

def split_dataset(input_path: str, output_dir: str, train_ratio=0.8, val_ratio=0.1, seed=42):
    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    random.seed(seed)
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = samples[:n_train]
    val = samples[n_train : n_train + n_val]
    test = samples[n_train + n_val :]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in [("train", train), ("val", val), ("test", test)]:
        with open(out / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Split {n} samples -> train:{len(train)} val:{len(val)} test:{len(test)}")

if __name__ == "__main__":
    import sys
    split_dataset(sys.argv[1], sys.argv[2])
