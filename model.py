import os
from dotenv import load_dotenv
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

load_dotenv()

MODEL_PATH = "models/trocr-finetuned" if os.getenv("USE_FINETUNED") == "true" else "microsoft/trocr-small-handwritten"


def load_model():
    processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
    return processor, model


def predict(image, processor, model):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
