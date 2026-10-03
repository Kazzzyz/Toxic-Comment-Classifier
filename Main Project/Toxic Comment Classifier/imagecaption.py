
import torch
from functools import lru_cache
from transformers import BlipProcessor, BlipForConditionalGeneration


@lru_cache(maxsize=1)
def load_blip():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "Salesforce/blip-image-captioning-base"

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)

    model.to(device)
    model.eval()

    return processor, model, device


def generate_caption(image):
    processor, model, device = load_blip()

    image = image.convert("RGB")

    inputs = processor(images=image,return_tensors="pt").to(device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=50)

    caption = processor.decode(output[0],skip_special_tokens=True)

    return caption.strip()