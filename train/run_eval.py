"""Load trained LoRA model and run evaluation on test set."""
import json, torch
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from rouge_score import rouge_scorer


def load_model(lora_path, base_model):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    processor = AutoProcessor.from_pretrained(base_model)
    return model, processor


def generate(model, processor, user_msg, image_path=None, max_new_tokens=256):
    content = []
    if image_path and Path(image_path).exists():
        content.append({"type": "image", "image": f"file://{image_path}"})
    text = user_msg.replace("<image>\n", "").replace("<image>", "")
    content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if image_path and Path(image_path).exists():
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=[text_input], images=[image], return_tensors="pt", padding=True)
    else:
        inputs = processor(text=[text_input], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


def extract_verdict(text):
    if "不合格" in text: return "不合格"
    if "合格" in text: return "合格"
    return "未知"


def main():
    base_model = "models/Qwen2-VL-2B-Instruct"
    lora_path = "outputs/lora_defect/best"
    test_path = "data/test.json"
    output_path = "outputs/eval_results.json"
    pred_cache = Path("outputs/eval_predictions.json")

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if pred_cache.exists():
        print("Loading cached predictions...", flush=True)
        with open(pred_cache, "r", encoding="utf-8") as f:
            cached = json.load(f)
        predictions = cached["predictions"]
        labels = cached["labels"]
        print(f"Loaded {len(predictions)} cached predictions.", flush=True)
    else:
        print("Loading model...", flush=True)
        model, processor = load_model(lora_path, base_model)
        print("Model loaded.", flush=True)
        print(f"Evaluating {len(test_data)} samples...", flush=True)
        predictions = []
        labels = []
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
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(test_data)} done", flush=True)
        with open(pred_cache, "w", encoding="utf-8") as f:
            json.dump({"predictions": predictions, "labels": labels}, f, ensure_ascii=False, indent=2)
        print("Predictions cached.", flush=True)

    pred_verdicts = [extract_verdict(p) for p in predictions]
    label_verdicts = [extract_verdict(l) for l in labels]
    acc = accuracy_score(label_verdicts, pred_verdicts)
    prec, rec, f1, _ = precision_recall_fscore_support(label_verdicts, pred_verdicts, average="weighted", zero_division=0)
    report = classification_report(label_verdicts, pred_verdicts, zero_division=0)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_scores = [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, labels)]
    rouge_l = round(sum(rouge_scores) / len(rouge_scores), 4)

    results = {
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4), "rouge_l": rouge_l,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== Evaluation Results ===", flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    print(f"\nDetailed report:\n{report}", flush=True)

    samples_out = []
    for i in range(min(10, len(test_data))):
        samples_out.append({
            "image": test_data[i].get("image", ""),
            "label": labels[i],
            "prediction": predictions[i],
        })
    with open("outputs/eval_samples.json", "w", encoding="utf-8") as f:
        json.dump(samples_out, f, ensure_ascii=False, indent=2)
    print(f"\nSample predictions saved to outputs/eval_samples.json", flush=True)


if __name__ == "__main__":
    main()
