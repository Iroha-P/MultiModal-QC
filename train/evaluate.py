import json
import time
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from rouge_score import rouge_scorer

def extract_verdict(text: str) -> str:
    if "不合格" in text:
        return "不合格"
    if "合格" in text:
        return "合格"
    return "未知"

def evaluate_classification(predictions: list[str], labels: list[str]) -> dict:
    pred_verdicts = [extract_verdict(p) for p in predictions]
    label_verdicts = [extract_verdict(l) for l in labels]
    acc = accuracy_score(label_verdicts, pred_verdicts)
    prec, rec, f1, _ = precision_recall_fscore_support(
        label_verdicts, pred_verdicts, average="weighted", zero_division=0)
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "report": classification_report(label_verdicts, pred_verdicts, zero_division=0),
    }

def evaluate_generation(predictions: list[str], references: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]
    return {"rouge_l": round(sum(scores) / len(scores), 4)}

def evaluate_latency(model, test_inputs: list[str], n_samples: int = 20) -> dict:
    times = []
    for inp in test_inputs[:n_samples]:
        start = time.time()
        model.chat(inp)
        times.append(time.time() - start)
    return {
        "avg_latency_ms": round(sum(times) / len(times) * 1000, 1),
        "throughput_per_sec": round(len(times) / sum(times), 2),
    }

def run_evaluation(test_data_path: str, model, output_path: str):
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    predictions = []
    labels = []
    for sample in test_data:
        user_msg = sample["conversations"][0]["content"]
        label = sample["conversations"][1]["content"]
        image = sample.get("image")
        video = sample.get("video")
        pred = model.chat(user_msg, image_path=image, video_path=video)
        predictions.append(pred)
        labels.append(label)
    cls_metrics = evaluate_classification(predictions, labels)
    gen_metrics = evaluate_generation(predictions, labels)
    results = {**cls_metrics, **gen_metrics}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Evaluation results saved to {output_path}")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results

if __name__ == "__main__":
    import sys
    print("Usage: python evaluate.py <test_data.json> <output.json>")
    print("Requires model to be loaded separately.")
