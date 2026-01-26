from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
import joblib
import time
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import secrets
import os


# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TIME_STEPS = 50
FUTURE_DAYS = 10

# APP
app = FastAPI(title="API de Previsão LSTM Avançada")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


API_KEYS = set()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "Modelo", "modelo_lstm.h5")
scaler_path = os.path.join(BASE_DIR, "..", "Modelo", "scaler.pkl")

model = load_model(model_path, custom_objects={"mse": MeanSquaredError()}, compile=False)
scaler = joblib.load(scaler_path)


# AUTHENTICATION
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="API Key inválida ou não registrada")


# MODELO JSON
class PriceRequest(BaseModel):
    prices: List[float]


# PREVISÃO MULTI-STEP
def predict_next_days(prices: np.ndarray, n_days=5):
    if len(prices) < TIME_STEPS:
        raise HTTPException(400, "São necessários pelo menos 50 preços")

    prices = prices.reshape(-1, 1)
    scaled = scaler.transform(prices)

    input_seq = scaled[-TIME_STEPS:]
    predictions = []

    for _ in range(n_days):
        X = input_seq.reshape(1, TIME_STEPS, 1)
        pred_scaled = model.predict(X, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)[0][0]
        predictions.append(round(float(pred), 2))

        input_seq = np.append(input_seq[1:], pred_scaled, axis=0)

    return predictions


# HEALTH
@app.get("/")
def health():
    return {"status": "API rodando", "modelo": "LSTM"}


# GERAR API KEY
@app.post("/generate_api_key")
def generate_api_key():
    api_key = secrets.token_hex(16)
    API_KEYS.add(api_key)
    logger.info(f"Nova API Key gerada")
    return {
        "api_key": api_key,
        "aviso": "Guarde esta chave. Para os outros endpoints."
    }

# JSON PREDICT
@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict_json(data: PriceRequest):
    start = time.time()
    prices = np.array(data.prices, dtype=float)

    preds = predict_next_days(prices, FUTURE_DAYS)

    return {
        "previsoes_5_dias": preds,
        "tempo_ms": round((time.time() - start) * 1000, 2)
    }

# CSV PREDICT
@app.post("/predict_csv", dependencies=[Depends(verify_api_key)])
def predict_csv(
    file: UploadFile = File(...),
    column_name: Optional[str] = None
):

    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Envie um arquivo CSV válido")

    df = pd.read_csv(file.file)

    if df.empty:
        raise HTTPException(400, "CSV vazio")

    if column_name is None:
        column_name = df.columns[0]

    if column_name not in df.columns:
        raise HTTPException(400, f"Coluna '{column_name}' não encontrada")

    prices = df[column_name].dropna().values.astype(float)

    start = time.time()
    preds = predict_next_days(prices, FUTURE_DAYS)

    return {
        "previsoes_10_dias": preds,
        "tempo_ms": round((time.time() - start) * 1000, 2)
    }
