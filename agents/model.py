from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch


class QwenVLModel:
    def __init__(self, model_path: str, lora_path: str | None = None, device: str = "cuda", use_4bit: bool = True):
        self.device = device
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, quantization_config=bnb_config, device_map="auto"
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto"
            )
        if lora_path and Path(lora_path).exists():
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    def chat(self, prompt: str, image_path: str | None = None, video_path: str | None = None,
             max_new_tokens: int = 1024) -> str:
        content = []
        if image_path and Path(image_path).exists():
            content.append({"type": "image", "image": f"file://{image_path}"})
        if video_path and Path(video_path).exists():
            content.append({"type": "video", "video": f"file://{video_path}", "fps": 1.0})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = output_ids[0][len(inputs.input_ids[0]):]
        return self.processor.decode(trimmed, skip_special_tokens=True)

    def switch_lora(self, lora_path: str):
        if hasattr(self.model, "load_adapter"):
            self.model.load_adapter(lora_path)
        else:
            from peft import PeftModel
            base = self.model.base_model if hasattr(self.model, "base_model") else self.model
            self.model = PeftModel.from_pretrained(base, lora_path)
