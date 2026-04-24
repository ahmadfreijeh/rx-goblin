from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_extractor():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model


def extract_fields(text, extractor):
    tokenizer, model = extractor

    prompt = f"""
    Extract the following fields from this prescription text and return them as key:value pairs.
    Fields: drug, dosage, frequency, prescriber
    Text: {text}
    """

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    fields = {"drug": None, "dosage": None, "frequency": None, "prescriber": None}
    for line in result.strip().split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().lower()
            if key in fields:
                fields[key] = value.strip()

    return fields
