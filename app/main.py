from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil, os
from pathlib import Path
from detect import DeepFakeDetector

BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

app = FastAPI(title="DeepFake Video Detector - TF")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

MODEL_PATH = BASE_DIR / "model" / "final_model.h5"
detector = DeepFakeDetector(str(MODEL_PATH))

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = file.filename
    temp_path = TEMP_DIR / filename
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        score = detector.predict_video(str(temp_path))
        result = "FAKE" if score > 0.7 else "REAL"
        response = {"prediction": result, "confidence": float(score)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        try: os.remove(temp_path)
        except: pass

    return JSONResponse(response)
