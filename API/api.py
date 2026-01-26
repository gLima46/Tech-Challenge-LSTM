from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
import joblib
import time
import logging
import tensorflow as tf  # Alterado para importar o TF inteiro
import secrets
import os

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =========================
# CONSTANTES
# =========================
TIME_STEPS = 50
FUTURE_DAYS = 10

# =========================
# APP
# =========================
app = FastAPI(title="API de Previsão LSTM Avançada")

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# AUTENTICAÇÃO
# =========================
API_KEYS = set()

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in API_KEYS:
        # Dica: Em produção, você pode querer ter uma chave "mestra" fixa nas variáveis de ambiente
        raise HTTPException(
            status_code=401,
            detail="API Key inválida ou não registrada"
        )

# =========================
# CAMINHOS DOS ARQUIVOS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ajuste os caminhos conforme a estrutura do seu container
model_path = os.path.join(BASE_DIR, "..", "Modelo", "modelo_lstm.h5")
scaler_path = os.path.join(BASE_DIR, "..", "Modelo", "scaler.pkl")

# =========================
# LOAD MODEL & SCALER
# =========================
try:
    # CORREÇÃO PRINCIPAL AQUI:
    # 1. Usamos tf.keras.models.load_model para garantir o uso do TF interno
    # 2. compile=False ignora erros de otimizadores e configs antigas (resolve o batch_shape)
    logger.info(f"Carregando modelo de: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    logger.info(f"Carregando scaler de: {scaler_path}")
    scaler = joblib.load(scaler_path)
    logger.info("Modelo e Scaler carregados com sucesso.")

except Exception as e:
    logger.error(f"Erro crítico ao carregar modelo ou scaler: {e}")
    # Opcional: Impedir a API de iniciar se não houver modelo
    # raise e 
    model = None
    scaler = None

# =========================
# MODELO DE ENTRADA
# =========================
class PriceRequest(BaseModel):
    prices: List[float]

# =========================
# FUNÇÃO DE PREVISÃO
# =========================
def predict_next_days(prices: np.ndarray, n_days: int):
    if model is None or scaler is None:
        raise HTTPException(500, "Modelo não carregado no servidor.")

    if len(prices) < TIME_STEPS:
        raise HTTPException(
            status_code=400,
            detail=f"São necessários pelo menos {TIME_STEPS} preços"
        )

    # Prepara os dados
    prices = prices.reshape(-1, 1)
    scaled = scaler.transform(prices)

    # Pega os últimos TIME_STEPS
    input_seq = scaled[-TIME_STEPS:]
    predictions = []

    for _ in range(n_days):
        # Reshape para (1, 50, 1) compatível com LSTM
        X = input_seq.reshape(1, TIME_STEPS, 1)
        
        # verbose=0 evita sujar o log
        pred_scaled = model.predict(X, verbose=0)
        
        # Desnormaliza
        pred = scaler.inverse_transform(pred_scaled)[0][0]
        predictions.append(round(float(pred), 2))

        # Atualiza a sequência de entrada removendo o primeiro e adicionando a nova previsão
        input_seq = np.append(input_seq[1:], pred_scaled, axis=0)

    return predictions

# =========================
# ENDPOINTS
# =========================
@app.get("/")
def health():
    status_modelo = "Ativo" if model is not None else "Erro no carregamento"
    return {"status": "API rodando", "modelo": status_modelo}

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

    # Se não informar coluna, pega a primeira
    if column_name is None:
        column_name = df.columns[0]

    if column_name not in df.columns:
        raise HTTPException(
            400,
            f"Coluna '{column_name}' não encontrada. Colunas disponíveis: {list(df.columns)}"
        )

    # Limpeza básica e conversão
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