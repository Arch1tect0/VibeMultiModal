"""Shared CLIP model loader and zero-shot scoring function."""

from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src import config

_clip_model = None
_clip_processor = None


def get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None or _clip_processor is None:
        _clip_processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL_NAME)
        _clip_model = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME).to(config.TORCH_DEVICE)
        _clip_model.eval()
    return _clip_model, _clip_processor


def zero_shot(image_path: Path, prompts: Dict[str, str]) -> Tuple[str, float, Dict[str, float]]:
    model, processor = get_clip()
    image = Image.open(image_path).convert("RGB")
    labels = list(prompts.keys())
    text_prompts = [prompts[label] for label in labels]
    inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding=True).to(config.TORCH_DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
    scores = {labels[i]: float(probs[i].item()) for i in range(len(labels))}
    best_label = max(scores, key=scores.get)
    return best_label, scores[best_label], scores
