import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from preprocessing import preprocess
from model import load_model, predict
from correction import load_drug_list, correct_drug_name
from extractor import load_extractor, extract_fields


@asynccontextmanager
async def lifespan(app):
    app.state.processor, app.state.model = load_model()
    app.state.drug_list = load_drug_list()
    app.state.extractor = load_extractor()
    yield


app = FastAPI(title="rxgoblin", description="Handwritten prescription OCR API", lifespan=lifespan)


def read_image(file_bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    image = read_image(await file.read())
    if image is None:
        return JSONResponse(status_code=400, content={"error": "could not read image"})
    lines = preprocess(image)
    text = " ".join([predict(line, app.state.processor, app.state.model) for line in lines])
    return JSONResponse(content={"num_lines": len(lines), "image_shape": list(image.shape), "text": text})


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    image = read_image(await file.read())
    lines = preprocess(image)
    text = " ".join([predict(line, app.state.processor, app.state.model) for line in lines])
    corrected = correct_drug_name(text, app.state.drug_list)
    fields = extract_fields(corrected, app.state.extractor)
    return JSONResponse(content=fields)
