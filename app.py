import io
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from fastapi.responses import FileResponse
from ultralytics import YOLO

app = FastAPI(title="Forklift & Pallet Detector")
model = YOLO("weights/best.pt")

@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = 0.25):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    result = model.predict(image, conf=conf, verbose=False)[0]
    detections = [
        {
            "class": model.names[int(box.cls)],
            "confidence": round(float(box.conf), 3),
            "bbox": [round(float(v), 1) for v in box.xyxy[0]],
        }
        for box in result.boxes
    ]
    return {"count": len(detections), "detections": detections} 