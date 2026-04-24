# rx-goblin

A handwritten prescription OCR API. Upload a prescription image, get back the drug name, dosage, frequency, and prescriber as structured JSON.

---

## What it does

```
Image → Preprocessing → TrOCR (OCR) → Drug Correction → Field Extraction → JSON
```

1. **Preprocessing** — deskews, binarizes, and splits the image into individual text lines
2. **TrOCR** — runs Microsoft's handwriting OCR model on each line
3. **Drug Correction** — fuzzy-matches the result against the FDA drug dictionary
4. **Field Extraction** — uses Flan-T5 to pull out structured fields from the corrected text

---

## Stack

| Layer | Tool |
|---|---|
| OCR Model | [TrOCR](https://huggingface.co/microsoft/trocr-small-handwritten) (HuggingFace) |
| Fine-tuning | LoRA via PEFT |
| Image processing | OpenCV + Pillow |
| Drug correction | rapidfuzz + FDA NDC dictionary |
| Field extraction | Flan-T5 (HuggingFace) |
| API | FastAPI |

---

## Setup

```bash
git clone https://github.com/your-username/rx-goblin.git
cd rx-goblin
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy the env file and configure:

```bash
cp .env.example .env
```

---

## Run the API

```bash
python main.py
```

> `main.py` calls uvicorn under the hood — no CLI flags needed.

---

## Endpoints

### `GET /health`
```json
{ "status": "ok" }
```

### `POST /ocr`
Upload an image, get back the raw OCR text.

```bash
curl -X POST http://localhost:8000/ocr -F "file=@prescription.jpg"
```

```json
{
  "num_lines": 4,
  "image_shape": [800, 600, 3],
  "text": "Amoxicillin 500mg twice daily"
}
```

### `POST /extract`
Full pipeline — returns structured fields.

```bash
curl -X POST http://localhost:8000/extract -F "file=@prescription.jpg"
```

```json
{
  "drug": "Amoxicillin",
  "dosage": "500mg",
  "frequency": "twice daily",
  "prescriber": null
}
```

---

## Fine-tuning (optional)

The fine-tuning pipeline is fully built using **LoRA (Low-Rank Adaptation)** via the PEFT library. The goal was to prove the concept of adapting a pretrained model to a new domain without retraining from scratch — training only ~0.47% of the model's parameters instead of all 62M.

The training script supports:
- LoRA adapters on the attention layers (`query`, `value`)
- CER and WER metrics evaluated after every epoch
- Gradient clipping to prevent exploding gradients
- Full config via `.env` (epochs, batch size, learning rate, sample count)
- Training loss/CER/WER graphs saved to `training_results.png`

**Note on dataset quality:** The Kaggle handwriting dataset used here contains single handwritten words (names). This caused the training loss to not decrease — the model had no meaningful signal to learn from. For fine-tuning to work properly, the dataset needs full handwritten text lines (e.g. the [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)). This is a classic **garbage in, garbage out** problem — the pipeline is correct, the data wasn't right for the task.

To run fine-tuning with a proper dataset:

1. Put your data in `data/processed/train/` and `data/processed/val/`  
   Each folder needs an `images/` directory and a `labels.csv` with `FILENAME` and `IDENTITY` columns.

2. Run training:

```bash
python train.py
```

3. Switch the API to use your fine-tuned model:

```bash
# in .env
USE_FINETUNED=true
```

Training config is controlled via `.env`:

```
TRAIN_EPOCHS=2
TRAIN_BATCH_SIZE=4
TRAIN_SAMPLES=5000
TRAIN_LR=1e-5
```

Results are saved to `training_results.png`.

---

## Datasets

| Dataset | Used for | Link |
|---|---|---|
| Handwriting Recognition (Kaggle) | Fine-tuning TrOCR on handwritten text | [kaggle.com](https://www.kaggle.com/datasets/landlord/handwriting-recognition?resource=download&select=validation_v2) |
| FDA National Drug Code Directory | Drug name fuzzy correction | [fda.gov](https://www.fda.gov/drugs/drug-approvals-and-databases/national-drug-code-directory) |

---

## Project structure

```
rx-goblin/
├── main.py             # Entry point — runs the server
├── api.py              # FastAPI app — routes and lifespan
├── preprocessing.py    # Image pipeline
├── model.py            # TrOCR load + predict
├── correction.py       # FDA drug name correction
├── extractor.py        # Flan-T5 field extraction
├── train.py            # LoRA fine-tuning
├── data/
│   ├── raw/fda/        # FDA drug dictionary
│   └── processed/      # Train/val splits
└── models/             # Fine-tuned checkpoints
```
