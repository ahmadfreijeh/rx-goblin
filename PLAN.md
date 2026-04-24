# rxgoblin — Project Plan

> Beginner-friendly learning project. Built in two versions — v1 gets something working fast using pre-trained models, v2 adds fine-tuning as an optional upgrade.

---

## System Overview

```
[Prescription Image]
        ↓
[Image Cleanup]           ← OpenCV: straighten, clean, cut into lines
        ↓
[Read Handwriting]        ← TrOCR: image → raw text
        ↓
[Fix Drug Names]          ← rapidfuzz: correct OCR typos against drug list
        ↓
[Structure the Output]    ← Flan-T5: extract fields as JSON
        ↓
[API Response]            ← FastAPI: { drug, dosage, frequency, prescriber }
```

This pipeline is **identical in v1 and v2**. The only difference is where the model comes from:

```
v1 → loads microsoft/trocr-base-handwritten from HuggingFace (pre-trained)
v2 → loads models/trocr-finetuned from local disk (your fine-tuned version)
```

Controlled by a single env var in `.env`:

```
USE_FINETUNED=false   # v1 — use pre-trained model
USE_FINETUNED=true    # v2 — use your fine-tuned checkpoint
```

In `model.py`:

```python
if os.getenv("USE_FINETUNED") == "true":
    model = VisionEncoderDecoderModel.from_pretrained("models/trocr-finetuned")
else:
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
```

---

## Version 1 — Pre-trained Only

### What you build
A working end-to-end pipeline using models exactly as they come from HuggingFace. No training, no datasets.

### What you learn
- OpenCV image processing
- How to load and use a pre-trained HuggingFace model
- Fuzzy string matching
- Building a REST API with FastAPI

### Phases

#### Phase 1 — Image Preprocessing
**File:** `preprocessing.py`  
**Goal:** Take a raw prescription photo → output clean text line images ready for the model.

Steps you implement one by one:
1. `deskew(image)` — straighten tilted document
2. `binarize(image)` — convert to black and white
3. `remove_noise(image)` — clean up spots and artifacts
4. `segment_lines(image)` — cut image into individual text lines
5. `normalize_line(line)` — resize each line to 32px height (what TrOCR expects)

Test in `notebooks/01_preprocessing.ipynb` — display each step visually.

**Core concept you learn:** How images are just arrays of numbers. Every function transforms that array.

---

#### Phase 2 — Handwriting Recognition
**File:** `model.py`  
**Goal:** Load TrOCR, feed it a line image, get text back.

```python
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def predict(image):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    output = model.generate(pixel_values)
    return processor.decode(output[0], skip_special_tokens=True)
```

Also implement the env var switch here so v2 just works later.

Test in `notebooks/02_trocr_baseline.ipynb` — show input image next to predicted text.

**Core concept you learn:** What a pre-trained model is. How tokenizers and processors work. What `generate()` does.

---

#### Phase 3 — Drug Name Correction
**File:** `correction.py`  
**Goal:** Fix OCR typos in drug names using fuzzy matching.

Steps:
1. Download FDA drug name list (free CSV from `open.fda.gov`)
2. Load it into a Python list
3. Use `rapidfuzz.process.extractOne()` to find the closest match

```python
from rapidfuzz import process

def correct_drug_name(raw_text, drug_list):
    match, score, _ = process.extractOne(raw_text, drug_list)
    return match if score > 80 else raw_text
```

**Core concept you learn:** Edit distance, fuzzy matching, why OCR output is never perfect.

---

#### Phase 4 — Structured Extraction
**File:** `extractor.py`  
**Goal:** Take corrected text → extract drug, dosage, frequency into JSON.

Use Flan-T5 with a simple prompt:

```python
from transformers import pipeline

extractor = pipeline("text2text-generation", model="google/flan-t5-base")

def extract_fields(text):
    prompt = f"Extract drug name, dosage, and frequency from: {text}"
    result = extractor(prompt, max_length=100)
    return result[0]["generated_text"]
```

**Core concept you learn:** Instruction-tuned models, prompt engineering, text2text generation.

---

#### Phase 5 — REST API
**File:** `api.py`  
**Goal:** Wrap the whole pipeline in two API endpoints.

```
POST /ocr      → upload image, returns raw text
POST /extract  → upload image, returns structured JSON
```

Wire all modules together:
`preprocessing.py` → `model.py` → `correction.py` → `extractor.py`

Test at `http://localhost:8000/docs` — FastAPI generates an interactive UI automatically.

**Core concept you learn:** How APIs work, request/response cycle, how to connect independent modules.

---

## Version 2 — Fine-tuned Model (Optional Upgrade)

> Only start this after v1 is fully working. Fine-tuning plugs into the same pipeline — it just creates a better model that replaces the pre-trained one.

### What you build
A fine-tuned version of TrOCR trained on handwriting data + synthetic prescription images. Saved to `models/trocr-finetuned/`. Switch to it by setting `USE_FINETUNED=true`.

### What you learn
- What fine-tuning actually is (vs training from scratch)
- What LoRA is and why it's preferred over full fine-tuning for small datasets
- How to load and format a dataset for a HuggingFace model
- What a training loop does
- How to measure model improvement (CER/WER)
- Experiment tracking with Weights & Biases

### Phases

#### Phase 6 — Datasets
**Files:** `generate_synthetic.py`, `data/`

**Dataset 1 — IAM Handwriting Database**
- 13,000+ English handwritten text line images with ground truth
- Register free at `fki.unibe.ch`, download `lines.tgz` + `ascii.tgz`
- Teaches the model diverse cursive handwriting

**Dataset 2 — Synthetic Prescription Data (you generate this)**
- Use TRDG to render drug names, abbreviations, dosages in cursive fonts
- Fonts: Dancing Script, Caveat (free Google Fonts)
- Target: ~3,000–5,000 images
- Run once: `python generate_synthetic.py`

**Dataset 3 — Real Prescription Images (stretch goal)**
- Kaggle Handwritten Prescription Dataset (~800 images)
- Free, needs Kaggle account

**Data split:**

| Split | % | Purpose |
|---|---|---|
| Train | 80% | Model learns from this |
| Validation | 10% | Check if improving during training |
| Test | 10% | Final score — never shown to model |

**Data folder:**
```
data/
├── raw/
│   ├── iam/              # IAM images + transcriptions
│   └── kaggle_rx/        # real prescription images
├── synthetic/
│   ├── images/
│   └── labels.csv
├── processed/
│   ├── train/
│   ├── val/
│   └── test/
```

---

#### Phase 7 — Fine-tuning with LoRA
**File:** `train.py`  
**Goal:** Fine-tune TrOCR on your data using LoRA, save new model to `models/trocr-finetuned/`.

**Why LoRA instead of full fine-tuning:**
- You're on Apple Silicon (MPS) — full fine-tuning of TrOCR is memory-heavy, LoRA trains ~1% of parameters so it fits comfortably
- Your dataset is small (~3-5k images) — full fine-tuning risks overfitting, LoRA is naturally more resistant
- Nearly same quality at 10% of the cost — good tradeoff for a portfolio project

**How LoRA works:**
Original model weights are frozen. LoRA injects small trainable matrices alongside them. Only those matrices get updated — the original model is never touched.

```
Original TrOCR weights  →  frozen, untouched
LoRA matrices (tiny)    →  these get trained
         ↓
models/trocr-finetuned/ →  saved, ready to swap in
```

**How to add LoRA (just a few lines on top of normal training):**
```python
from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(
    r=16,               # rank — controls how many parameters to train
    lora_alpha=32,      # scaling factor
    target_modules=["query", "value"],  # which layers to apply LoRA to
)
model = get_peft_model(model, lora_config)
# rest of training code stays exactly the same
```

Uses HuggingFace `Seq2SeqTrainer` — handles the training loop for you.  
Tracks loss and CER/WER with Weights & Biases (free account).

After training, set `USE_FINETUNED=true` in `.env` — the same pipeline now uses your improved model.

**Core concept you learn:** What fine-tuning is. How LoRA reduces memory and overfitting. What loss means. How to read a training curve.

---

#### Phase 8 — Evaluation
**File:** `evaluate.py`  
**Goal:** Measure how much fine-tuning improved things.

| Metric | What it measures | Target |
|---|---|---|
| CER (Character Error Rate) | % of characters wrong | Below 15% |
| WER (Word Error Rate) | % of words wrong | Below 25% |

Run: `python evaluate.py` → prints a before/after comparison table.

**Core concept you learn:** How to measure model quality properly. Why "accuracy" alone is not enough for OCR.

---

## Folder Structure

```
rx-goblin/
├── preprocessing.py          # Phase 1 — image cleanup
├── model.py                  # Phase 2 — TrOCR load + predict (+ env var switch)
├── correction.py             # Phase 3 — drug name fuzzy correction
├── extractor.py              # Phase 4 — Flan-T5 structured extraction
├── api.py                    # Phase 5 — FastAPI endpoints
├── train.py                  # Phase 7 — fine-tuning script (v2 only)
├── generate_synthetic.py     # Phase 6 — synthetic data generation (v2 only)
├── evaluate.py               # Phase 8 — CER/WER evaluation (v2 only)
├── .env                      # USE_FINETUNED=false
├── data/                     # datasets (v2 only)
├── models/                   # fine-tuned checkpoint saved here (v2 only)
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_trocr_baseline.ipynb
│   ├── 03_correction.ipynb
│   └── 04_finetuning.ipynb   # v2 only
├── requirements.txt
└── README.md
```

---

## Dependencies (requirements.txt)

```
torch
transformers
peft
Pillow
opencv-python
rapidfuzz
pyspellchecker
fastapi
uvicorn
python-multipart
python-dotenv
jiwer
trdg
datasets
wandb
```

---

## What to Show in Portfolio

**v1 (minimum):**
- Notebook showing prescription image → extracted text
- FastAPI `/docs` screenshot with both endpoints working
- Short demo: upload image, show JSON output

**v2 (bonus):**
- Before/after CER/WER table (pre-trained vs fine-tuned)
- W&B training curve screenshot (loss going down)
- Both modes working via `USE_FINETUNED` toggle
