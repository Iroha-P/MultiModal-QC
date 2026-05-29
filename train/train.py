import json
import sys
import yaml
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from PIL import Image


class VLMQADataset(Dataset):
    def __init__(self, qa_path: str, processor, max_length: int = 2048):
        with open(qa_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = sample.get("image")
        user_msg = sample["conversations"][0]["content"]
        assistant_msg = sample["conversations"][1]["content"]

        content = []
        if image_path and Path(image_path).exists():
            content.append({"type": "image", "image": f"file://{image_path}"})
        text_content = user_msg.replace("<image>\n", "").replace("<image>", "")
        content.append({"type": "text", "text": text_content})

        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_msg}]},
        ]
        return messages, image_path


def collate_fn(batch, processor, max_length):
    all_messages = [b[0] for b in batch]
    all_images = []

    for messages, image_path in batch:
        if image_path and Path(image_path).exists():
            all_images.append(Image.open(image_path).convert("RGB"))

    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in all_messages]

    if all_images:
        inputs = processor(
            text=texts, images=all_images, padding=True,
            truncation=True, max_length=max_length, return_tensors="pt"
        )
    else:
        inputs = processor(
            text=texts, padding=True,
            truncation=True, max_length=max_length, return_tensors="pt"
        )

    inputs["labels"] = inputs["input_ids"].clone()
    return inputs


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_val_loss, output_dir):
    ckpt_dir = Path(output_dir) / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir / "lora"))
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }, str(ckpt_dir / "train_state.pt"))
    print(f"Checkpoint saved: epoch={epoch+1}, step={global_step}", flush=True)


def load_checkpoint(model, optimizer, scheduler, output_dir):
    ckpt_dir = Path(output_dir) / "checkpoint"
    state_path = ckpt_dir / "train_state.pt"
    lora_path = ckpt_dir / "lora"
    if not state_path.exists() or not lora_path.exists():
        return 0, 0, float("inf")
    state = torch.load(str(state_path), map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    print(f"Resumed from checkpoint: epoch={state['epoch']+1}, step={state['global_step']}", flush=True)
    return state["epoch"], state["global_step"], state["best_val_loss"]


def train(config_path: str, resume: bool = False):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_path = cfg["model_id_or_path"]
    output_dir = cfg.get("output_dir", "outputs/lora_defect")
    num_epochs = cfg.get("num_train_epochs", 3)
    batch_size = cfg.get("batch_size", 1)
    grad_accum = cfg.get("gradient_accumulation_steps", 8)
    lr = cfg.get("learning_rate", 1e-4)
    warmup_ratio = cfg.get("warmup_ratio", 0.1)
    max_length = cfg.get("max_length", 2048)
    logging_steps = cfg.get("logging_steps", 10)

    print(f"Loading model from {model_path}...", flush=True)
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, quantization_config=bnb_config, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print("Model loaded.", flush=True)

    if cfg.get("freeze_vit", True):
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad = False
    print("Vision encoder frozen.", flush=True)

    ckpt_lora = Path(output_dir) / "checkpoint" / "lora"
    if resume and ckpt_lora.exists():
        print("Loading LoRA from checkpoint...", flush=True)
        model = PeftModel.from_pretrained(model, str(ckpt_lora), is_trainable=True)
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.get("lora_rank", 64),
            lora_alpha=cfg.get("lora_alpha", 128),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("LoRA initialized.", flush=True)

    train_ds = VLMQADataset(cfg["dataset_path"], processor, max_length)
    val_ds = VLMQADataset(cfg["val_dataset_path"], processor, max_length) if cfg.get("val_dataset_path") else None

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds) if val_ds else 0}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
    )
    total_steps = (len(train_ds) // batch_size // grad_accum) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if resume:
        start_epoch, global_step, best_val_loss = load_checkpoint(model, optimizer, scheduler, output_dir)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        processed = 0

        torch.manual_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        optimizer.zero_grad()

        steps_per_epoch = len(train_ds) // grad_accum
        start_sample = 0
        if epoch == start_epoch and global_step > 0:
            completed_in_epoch = global_step - (start_epoch * steps_per_epoch)
            start_sample = completed_in_epoch * grad_accum
            if start_sample > 0:
                print(f"Skipping {start_sample} samples (resuming mid-epoch)", flush=True)

        for i, idx in enumerate(indices):
            if i < start_sample:
                continue
            try:
                messages, image_path = train_ds[idx]
                inputs = collate_fn([(messages, image_path)], processor, max_length)
                inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

                outputs = model(**inputs)
                loss = outputs.loss / grad_accum
                loss.backward()
                epoch_loss += outputs.loss.item()
                processed += 1

                if (i + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % logging_steps == 0:
                        avg = epoch_loss / processed
                        print(f"Epoch {epoch+1}/{num_epochs} Step {global_step} Loss: {avg:.4f} LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)

            except Exception as e:
                print(f"Skipping sample {idx}: {e}")
                continue

        avg_train_loss = epoch_loss / max(processed, 1)
        print(f"Epoch {epoch+1} done. Avg train loss: {avg_train_loss:.4f}", flush=True)

        if val_ds:
            model.eval()
            val_loss = 0
            val_count = 0
            with torch.no_grad():
                for idx in range(len(val_ds)):
                    try:
                        messages, image_path = val_ds[idx]
                        inputs = collate_fn([(messages, image_path)], processor, max_length)
                        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
                        outputs = model(**inputs)
                        val_loss += outputs.loss.item()
                        val_count += 1
                    except Exception:
                        continue
            avg_val = val_loss / max(val_count, 1)
            print(f"Epoch {epoch+1} val loss: {avg_val:.4f}", flush=True)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                model.save_pretrained(f"{output_dir}/best")
                processor.save_pretrained(f"{output_dir}/best")
                print(f"Best model saved (val_loss={avg_val:.4f})", flush=True)

        model.save_pretrained(f"{output_dir}/epoch_{epoch+1}")
        save_checkpoint(model, optimizer, scheduler, epoch, global_step, best_val_loss, output_dir)
        print(f"Epoch checkpoint saved: {output_dir}/epoch_{epoch+1}", flush=True)

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Training complete. Final model saved to {output_dir}", flush=True)


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "train/configs/lora_defect.yaml"
    resume = "--resume" in sys.argv
    train(config, resume=resume)
