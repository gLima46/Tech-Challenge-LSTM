import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
import joblib
import time
import logging
import secrets
import keras 

os.environ["KERAS_BACKEND"] = "tensorflow"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


TIME_STEPS = 50
FUTURE_DAYS = 10


app = FastAPI(title="API de Previsão LSTM Avançada")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


API_KEYS = set()

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="API Key inválida ou não registrada"
        )


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "Modelo", "modelo_lstm.h5")
scaler_path = os.path.join(BASE_DIR, "..", "Modelo", "scaler.pkl")


model = None
scaler = None

try:
    logger.info(f"Versão do Keras: {keras.__version__}")
    logger.info(f"Tentando carregar modelo de: {model_path}")
    
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path, compile=False)
        logger.info("Modelo carregado com sucesso!")
    else:
        logger.error(f"ARQUIVO DE MODELO NÃO ENCONTRADO EM: {model_path}")

    logger.info(f"Tentando carregar scaler de: {scaler_path}")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        logger.info("Scaler carregado com sucesso!")
    else:
        logger.error(f"ARQUIVO DE SCALER NÃO ENCONTRADO EM: {scaler_path}")

except Exception as e:
    logger.error(f"Erro CRÍTICO ao carregar recursos: {str(e)}")



class PriceRequest(BaseModel):
    prices: List[float]


def predict_next_days(prices: np.ndarray, n_days: int):
    if model is None or scaler is None:
        raise HTTPException(
            status_code=500, 
            detail="O modelo de IA não foi carregado corretamente no servidor. Verifique os logs."
        )

    if len(prices) < TIME_STEPS:
        raise HTTPException(
            status_code=400,
            detail=f"São necessários pelo menos {TIME_STEPS} preços históricos para realizar a previsão."
        )

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

@app.get("/")
def health():
    status_modelo = "Ativo" if model is not None else "INATIVO (Erro ao carregar)"
    return {
        "status": "API online", 
        "modelo": status_modelo,
        "keras_version": keras.__version__
    }

@app.post("/generate_api_key")
def generate_api_key():
    api_key = secrets.token_hex(16)
    API_KEYS.add(api_key)
    logger.info("Nova API Key gerada")
    return {
        "api_key": api_key,
        "aviso": "Guarde esta chave. Ela será necessária para os outros endpoints."
    }

@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict_json(data: PriceRequest):
    start = time.time()
    prices = np.array(data.prices, dtype=float)

    preds = predict_next_days(prices, FUTURE_DAYS)

    return {
        "previsoes_10_dias": preds,
        "tempo_ms": round((time.time() - start) * 1000, 2)
    }

@app.post("/predict_csv", dependencies=[Depends(verify_api_key)])
def predict_csv(
    file: UploadFile = File(...),
    column_name: Optional[str] = None
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Envie um arquivo CSV válido")

    try:
        df = pd.read_csv(file.file)
    except Exception:
        raise HTTPException(400, "Erro ao ler o arquivo CSV.")

    if df.empty:
        raise HTTPException(400, "CSV vazio")

    if column_name is None:
        column_name = df.columns[0]

    if column_name not in df.columns:
        raise HTTPException(
            400,
            f"Coluna '{column_name}' não encontrada. Colunas disponíveis: {list(df.columns)}"
        )

    try:
        prices = df[column_name].dropna().values.astype(float)
    except ValueError:
        raise HTTPException(400, "A coluna selecionada contém dados não numéricos.")

    start = time.time()
    preds = predict_next_days(prices, FUTURE_DAYS)

    return {
        "previsoes_10_dias": preds,
        "tempo_ms": round((time.time() - start) * 1000, 2)
    }