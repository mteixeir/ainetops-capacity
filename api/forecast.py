from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from prophet import Prophet
import pandas as pd
import numpy as np
import json
import os
from openai import OpenAI
from datetime import datetime, timedelta

app = FastAPI()

# ── Servir archivos estáticos (HTML, CSS (Cascading Style Sheets), JS) ──
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")


# ── Generar datos sintéticos de un router por 24 meses ──
def generar_datos_sinteticos():
    fechas = pd.date_range(end=datetime.today(), periods=24, freq="MS")
    np.random.seed(42)
    tendencia = np.linspace(0, 30, 24)          # crecimiento gradual
    ruido     = np.random.normal(0, 2, 24)       # variación aleatoria realista

    df = pd.DataFrame({
        "ds": fechas,
        "cpu":    np.clip(40 + tendencia + ruido,              0, 100),
        "ram":    np.clip(50 + tendencia * 1.2 + ruido * 0.8,  0, 100),
        "trafico": np.clip(30 + tendencia * 1.5 + ruido * 1.2, 0, 100),
    })
    return df


# ── Correr Prophet (modelo de forecasting de series temporales) para una métrica ──
def forecast_metrica(df: pd.DataFrame, col: str, periodos: int = 12):
    data = df[["ds", col]].rename(columns={col: "y"})
    m = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                daily_seasonality=False, interval_width=0.95)
    m.fit(data)
    futuro  = m.make_future_dataframe(periods=periodos, freq="MS")
    pred    = m.predict(futuro)
    return pred[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periodos + len(df))


# ── Endpoint principal: recibe CSV (Comma-Separated Values) o usa datos sintéticos ──
@app.post("/api/forecast")
async def run_forecast(
    file: UploadFile = File(None),
    usar_sintetico: str = Form("true")
):
    # 1. Cargar datos
    if usar_sintetico == "true" or file is None:
        df = generar_datos_sinteticos()
    else:
        contenido = await file.read()
        from io import StringIO
        df = pd.read_csv(StringIO(contenido.decode("utf-8")), parse_dates=["ds"])
        for col in ["cpu", "ram", "trafico"]:
            if col not in df.columns:
                return {"error": f"El CSV debe tener columnas: ds, cpu, ram, trafico"}

    # 2. Forecast por métrica
    resultados = {}
    fechas_colapso = {}
    for metrica in ["cpu", "ram", "trafico"]:
        pred = forecast_metrica(df, metrica, periodos=12)
        resultados[metrica] = pred.to_dict(orient="records")

        # Buscar primera fecha donde la proyección supera 90% (umbral crítico)
        futuro = pred[pred["ds"] > df["ds"].max()]
        colapso = futuro[futuro["yhat"] >= 90]
        if not colapso.empty:
            fechas_colapso[metrica] = colapso.iloc[0]["ds"].strftime("%B %Y")
        else:
            fechas_colapso[metrica] = "No proyectado en 12 meses"

    # 3. Generar resumen ejecutivo con GPT (Generative Pre-trained Transformer)-4o-mini
    api_key = os.environ.get("OPENAI_API_KEY")
    resumen_ia = ""
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            prompt = f"""
Eres un arquitecto de redes senior presentando a la gerencia de una empresa de telecomunicaciones.
Basado en el análisis de forecasting (pronóstico) del router principal con proyección a 12 meses:

- CPU (Unidad Central de Procesamiento): colapso proyectado en {fechas_colapso.get('cpu')}
- RAM (Memoria de Acceso Aleatorio): colapso proyectado en {fechas_colapso.get('ram')}
- Tráfico de red: colapso proyectado en {fechas_colapso.get('trafico')}

Redacta un párrafo ejecutivo conciso (máximo 120 palabras) que:
1. Describa la situación de riesgo claramente
2. Mencione las fechas críticas
3. Recomiende la acción (reemplazo del equipo)
4. Use lenguaje profesional pero comprensible para gerencia no técnica
Responde solo en español.
"""
            respuesta = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            resumen_ia = respuesta.choices[0].message.content
        except Exception as e:
            resumen_ia = f"Error al conectar con OpenAI (Plataforma de IA): {str(e)}"
    else:
        resumen_ia = "⚠️ Configura OPENAI_API_KEY (Llave de API de OpenAI) en las variables de entorno para activar el resumen ejecutivo con IA."

    # 4. Datos históricos para el gráfico
    historico = df[["ds", "cpu", "ram", "trafico"]].copy()
    historico["ds"] = historico["ds"].astype(str)

    return {
        "historico": historico.to_dict(orient="records"),
        "forecast":  {k: [
            {**r, "ds": str(r["ds"])} for r in v
        ] for k, v in resultados.items()},
        "fechas_colapso": fechas_colapso,
        "resumen_ia": resumen_ia
    }