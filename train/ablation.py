"""Ablation experiments: LoRA rank and data size."""
import argparse
import copy
import json
import random
import sys
import tempfile
import time
import yaml
from pathlib import Path

BASE_CONFIG = "train/configs/lora_defect.yaml"
BASE_MODEL = "models/Qwen2-VL-2B-Instruct"
TEST_PATH = "data/test.json"
EXISTING_EVAL = "outputs/eval_results.json"
ABLATION_DIR = Path("outputs/ablation")

RANK_EXPERIMENTS = [
    {"rank": 16, "alpha": 32, "name": "rank_16"},
    {"rank": 32, "alpha": 64, "name": "rank_32"},
    {"rank": 128, "alpha": 256, "name": "rank_128"},
]

DATA_EXPERIMENTS = [
    {"ratio": 0.25, "name": "data_25"},
    {"ratio": 0.50, "name": "data_50"},
    {"ratio": 0.75, "name": "data_75"},
]


def load_base_config():
    with open(BASE_CONFIG, "r") as f:
        return yaml.safe_load(f)


def make_subset(train_path, ratio, seed=42):
    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    random.seed(seed)
    k = int(len(data) * ratio)
    subset = random.sample(data, k)
    out_path = str(ABLATION_DIR / f"train_subset_{int(ratio*100)}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False)
    print(f"Created subset: {k}/{len(data)} samples -> {out_path}", flush=True)
    return out_path, k


def run_train(cfg, output_dir):
    cfg_path = str(ABLATION_DIR / "tmp_config.yaml")
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    print(f"\n{'='*60}", flush=True)
    print(f"Training: {output_dir}", flush=True)
    print(f"  rank={cfg.get('lora_rank')}, alpha={cfg.get('lora_alpha')}", flush=True)
    print(f"  data={cfg['dataset_path']}", flush=True)
    print(f"{'='*60}", flush=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location("train_module", "train/train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.train(cfg_path, resume=False)


def run_eval(lora_path, output_dir):
    print(f"\nEvaluating: {lora_path}", flush=True)
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_module", "train/run_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    load_model, generate, extract_verdict = mod.load_model, mod.generate, mod.extract_verdict

    model, processor = load_model(lora_path, BASE_MODEL)

    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    predictions, labels = [], []
    for i, sample in enumerate(test_data):
        user_msg = sample["conversations"][0]["content"]
        label = sample["conversations"][1]["content"]
        image = sample.get("image")
        try:
            pred = generate(model, processor, user_msg, image)
        except Exception as e:
            print(f"  Sample {i} error: {e}", flush=True)
            pred = ""
        predictions.append(pred)
        labels.append(label)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(test_data)} done", flush=True)

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from rouge_score import rouge_scorer

    pred_v = [extract_verdict(p) for p in predictions]
    label_v = [extract_verdict(l) for l in labels]
    acc = accuracy_score(label_v, pred_v)
    prec, rec, f1, _ = precision_recall_fscore_support(label_v, pred_v, average="weighted", zero_division=0)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l = sum(scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, labels)) / len(labels)

    results = {
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4), "rouge_l": round(rouge_l, 4),
    }

    out_file = Path(output_dir) / "eval_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results: {json.dumps(results)}", flush=True)

    import gc, torch
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def get_existing_results():
    if Path(EXISTING_EVAL).exists():
        with open(EXISTING_EVAL, "r") as f:
            return json.load(f)
    return None


def run_rank_ablation():
    all_results = []

    existing = get_existing_results()
    if existing:
        entry = {"rank": 64, "alpha": 128, **existing}
        all_results.append(entry)
        print(f"Reusing rank=64 results: {existing}", flush=True)

    for exp in RANK_EXPERIMENTS:
        output_dir = str(ABLATION_DIR / exp["name"])
        best_path = f"{output_dir}/best"

        if Path(f"{output_dir}/eval_results.json").exists():
            with open(f"{output_dir}/eval_results.json", "r") as f:
                results = json.load(f)
            print(f"Skipping {exp['name']} (already done): {results}", flush=True)
        else:
            cfg = load_base_config()
            cfg["lora_rank"] = exp["rank"]
            cfg["lora_alpha"] = exp["alpha"]
            cfg["output_dir"] = output_dir
            run_train(cfg, output_dir)

            lora_path = best_path if Path(best_path).exists() else output_dir
            results = run_eval(lora_path, output_dir)

        entry = {"rank": exp["rank"], "alpha": exp["alpha"], **results}
        all_results.append(entry)

    all_results.sort(key=lambda x: x["rank"])
    save_path = ABLATION_DIR / "rank_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nRank ablation complete. Results saved to {save_path}", flush=True)
    print_table(all_results, "rank")
    return all_results


def run_data_ablation():
    all_results = []

    existing = get_existing_results()
    if existing:
        entry = {"ratio": 1.0, "samples": 1380, **existing}
        all_results.append(entry)
        print(f"Reusing data=100% results: {existing}", flush=True)

    cfg_base = load_base_config()
    train_path = cfg_base["dataset_path"]

    for exp in DATA_EXPERIMENTS:
        output_dir = str(ABLATION_DIR / exp["name"])

        if Path(f"{output_dir}/eval_results.json").exists():
            with open(f"{output_dir}/eval_results.json", "r") as f:
                results = json.load(f)
            print(f"Skipping {exp['name']} (already done): {results}", flush=True)
            subset_count = int(1380 * exp["ratio"])
        else:
            subset_path, subset_count = make_subset(train_path, exp["ratio"])
            cfg = copy.deepcopy(cfg_base)
            cfg["dataset_path"] = subset_path
            cfg["output_dir"] = output_dir
            run_train(cfg, output_dir)

            best_path = f"{output_dir}/best"
            lora_path = best_path if Path(best_path).exists() else output_dir
            results = run_eval(lora_path, output_dir)

        entry = {"ratio": exp["ratio"], "samples": int(1380 * exp["ratio"]), **results}
        all_results.append(entry)

    all_results.sort(key=lambda x: x["ratio"])
    save_path = ABLATION_DIR / "data_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nData ablation complete. Results saved to {save_path}", flush=True)
    print_table(all_results, "ratio")
    return all_results


def print_table(results, key):
    print(f"\n{'='*60}", flush=True)
    print(f"  {key:>8}  Acc     F1      Prec    Recall  ROUGE-L", flush=True)
    print(f"  {'-'*55}", flush=True)
    for r in results:
        val = r[key]
        if isinstance(val, float) and val <= 1:
            val = f"{val:.0%}"
        print(f"  {str(val):>8}  {r['accuracy']:.4f}  {r['f1']:.4f}  {r['precision']:.4f}  {r['recall']:.4f}  {r['rouge_l']:.4f}", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["rank", "data", "all"], default="all")
    args = parser.parse_args()

    start = time.time()
    if args.exp in ("rank", "all"):
        run_rank_ablation()
    if args.exp in ("data", "all"):
        run_data_ablation()
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed/60:.1f} minutes", flush=True)
