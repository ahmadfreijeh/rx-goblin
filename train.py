import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv
from torch.utils.data import Dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import get_peft_model, LoraConfig
from jiwer import cer, wer

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "microsoft/trocr-small-handwritten"
OUTPUT_DIR = "models/trocr-finetuned"
DATA_DIR = "data/processed"
EPOCHS = int(os.getenv("TRAIN_EPOCHS", 2))
BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", 4))
LEARNING_RATE = float(os.getenv("TRAIN_LR", 5e-5))
TRAIN_SAMPLES = int(os.getenv("TRAIN_SAMPLES", 5000))


# ── Dataset ───────────────────────────────────────────────────────────────────

class PrescriptionDataset(Dataset):
    def __init__(self, split, processor):
        # load labels.csv — columns: image_path, text
        csv_path = os.path.join(DATA_DIR, split, "labels.csv")
        self.df = pd.read_csv(csv_path).dropna(subset=["IDENTITY"]).head(TRAIN_SAMPLES)
        self.processor = processor
        self.split_dir = os.path.join(DATA_DIR, split, "images")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # load image and convert to RGB
        image = Image.open(os.path.join(self.split_dir, row["FILENAME"])).convert("RGB")

        # convert image to pixel values the model understands
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        # tokenize the ground truth text
        labels = self.processor.tokenizer(
            str(row["IDENTITY"]),
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()

        # replace padding token id with -100 so loss ignores padding
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(pred, processor):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # predictions can come as a tuple (ids, past_key_values, ...) — take only the ids
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    # replace -100 back to pad token for decoding
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # decode predictions and labels to text
    pred_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_texts = processor.batch_decode(label_ids, skip_special_tokens=True)

    return {
        "cer": cer(label_texts, pred_texts),
        "wer": wer(label_texts, pred_texts),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    print("Loading processor and model...")
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    # set required model config for seq2seq generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # apply LoRA — only trains ~1% of parameters
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=16,                           # rank — how many parameters to train
        lora_alpha=32,                  # scaling factor
        target_modules=["query", "value"],  # which layers to apply LoRA to
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # shows how few parameters we're training

    # load datasets
    print("Loading datasets...")
    train_dataset = PrescriptionDataset("train", processor)
    val_dataset = PrescriptionDataset("val", processor)
    print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

    # training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        evaluation_strategy="epoch",    # evaluate after every epoch
        save_strategy="epoch",          # save checkpoint after every epoch
        load_best_model_at_end=True,    # keep the best checkpoint
        predict_with_generate=True,     # use generate() during evaluation
        logging_steps=50,
        report_to="none",
        max_grad_norm=1.0,              # clip gradients to prevent explosion
    )

    # trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
    )

    print("Starting training...")
    trainer.train()

    # save final model
    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Done.")
    plot_training(trainer)


def plot_training(trainer):
    logs = trainer.state.log_history

    epochs, train_loss, val_cer, val_wer = [], [], [], []

    for log in logs:
        if "loss" in log:
            train_loss.append(log["loss"])
            epochs.append(log["epoch"])
        if "eval_cer" in log:
            val_cer.append(log["eval_cer"])
        if "eval_wer" in log:
            val_wer.append(log["eval_wer"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, train_loss, marker="o")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(val_cer, marker="o", color="orange")
    axes[1].set_title("Validation CER")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("CER")

    axes[2].plot(val_wer, marker="o", color="green")
    axes[2].set_title("Validation WER")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("WER")

    plt.tight_layout()
    plt.savefig("training_results.png")
    print("Graphs saved to training_results.png")


if __name__ == "__main__":
    train()
