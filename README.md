Tech Challenge – Fase 04
Pós Tech – Machine Learning Engineering


🎯 Objetivo
Desenvolver um modelo de Deep Learning utilizando LSTM (Long Short-Term Memory) para prever o preço de fechamento de ações, realizando todo o pipeline de Machine Learning, desde a coleta dos dados até o deploy do modelo em uma API RESTful.


📊 Coleta de Dados
Os dados históricos de preços de ações foram coletados utilizando a biblioteca yfinance, com foco no preço de fechamento (Close).
Fonte: Yahoo Finance
Frequência: Diária
Período: 2014-01-01 a 2024-12-31


🔧 Pré-processamento
As seguintes etapas foram aplicadas:
Seleção da coluna Close
Normalização dos dados com MinMaxScaler
Criação de janelas temporais (sliding window) para séries temporais
Separação em conjuntos de treino e teste


🧠 Modelo LSTM
O modelo foi construído utilizando TensorFlow/Keras, com a seguinte abordagem:
Camadas LSTM para captura de padrões temporais
Camadas Dense para regressão
Função de perda: Mean Squared Error (MSE)
Otimizador: Adam


📈 Avaliação do Modelo
O modelo foi avaliado utilizando métricas de regressão:
MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
Essas métricas permitem avaliar o erro médio absoluto e o impacto de grandes desvios nas previsões.


💾 Salvamento do Modelo
Após o treinamento, os seguintes artefatos foram salvos:
modelo_lstm.h5 – Modelo treinado
scaler.pkl – Scaler utilizado na normalização
Esses arquivos são utilizados posteriormente para inferência na API.


🚀 Deploy da API
O modelo foi disponibilizado através de uma API RESTful desenvolvida com FastAPI, permitindo previsões de preços futuros a partir de dados históricos.
Endpoints disponíveis:
GET / – Health check da API
POST /generate_api_key – Geração de chave de acesso
POST /predict – Previsão via JSON
POST /predict_csv – Previsão via arquivo CSV
A API suporta:
Autenticação por API Key
Previsão multi-step
Entrada via JSON ou CSV
Medição do tempo de resposta


📊 Monitoramento
Foram implementados:
Logging de eventos
Medição do tempo de inferência
Controle de acesso via API Key


🐳 Containerização
A aplicação foi containerizada utilizando Docker, facilitando o deploy e a escalabilidade em ambientes de produção.

https://tech-challenge-lstm.onrender.com/docs#/default/

📦 Tecnologias Utilizadas
Python
TensorFlow / Keras
FastAPI
Pandas / NumPy
Scikit-learn
Docker