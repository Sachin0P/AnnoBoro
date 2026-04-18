import torch
import numpy as np
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class Captioner:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[Captioner] Loading BLIP on device: {self.device}")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def caption(self, frame: np.ndarray) -> str:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        inputs = self.processor(pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=50)
        return self.processor.decode(output[0], skip_special_tokens=True)
