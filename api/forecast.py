from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
import numpy as np
import os
import httpx
from datetime import datetime
from io import StringIO

app = FastAPI()

# ── Servir archivos estáticos (HTML, CSS, JS) ──
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")


# ── Generar datos sintéticos de un router por 24 meses ──
def generar_datos_sinteticos():
    fechas = pd.date_range(end=datetime.today(), periods=24, freq="MS")
    np.random.seed(42)
    tendencia = np.linspace(0, 30, 24)
    ruido     = np.random.normal(0, 2, 24)
    df = pd.DataFrame({
        "ds":      fechas,
        "cpu":     np.clip(40 + tendencia + ruido,              0, 100),
        "ram":     np.clip(50 + tendencia * 1.2 + ruido * 0.8,  0, 100),
        "trafico": np.clip(30 + tendencia * 1.5 + ruido * 1.2,  0, 100),
    })
    return df


# ── Forecast con Holt-Winters (Suavizamiento Exponencial Triple) ──
def forecast_metrica(df: pd.DataFrame, col: str, periodos: int = 12):
    serie = df[col].values.astype(float)

    modelo = ExponentialSmoothing(serie, trend="add", seasonal=None)
    ajuste = modelo.fit(optimized=True)
    pred   = ajuste.forecast(periodos)

    residuos = ajuste.resid
    std      = float(np.std(residuos))

    fechas_hist   = pd.date_range(df["ds"].iloc[0], periods=len(serie), freq="MS")
    fechas_futuro = pd.date_range(df["ds"].iloc[-1] + pd.DateOffset(months=1),
                                  periods=periodos, freq="MS")

    hist_rows = [{"ds": str(d.date()), "yhat": round(float(v), 2),
                  "yhat_lower": round(float(v), 2),
                  "yhat_upper": round(float(v), 2)}
                 for d, v in zip(fechas_hist, serie)]

    pred_rows = [{"ds": str(d.date()),
                  "yhat":       round(np.clip(float(v), 0, 100), 2),
                  "yhat_lower": round(np.clip(float(v) - 1.96 * std, 0, 100), 2),
                  "yhat_upper": round(np.clip(float(v) + 1.96 * std, 0, 100), 2)}
                 for d, v in zip(fechas_futuro, pred)]

    return hist_rows + pred_rows, fechas_futuro, pred


# ── Endpoint principal ──
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
        df = pd.read_csv(StringIO(contenido.decode("utf-8")), parse_dates=["ds"])
        for col in ["cpu", "ram", "trafico"]:
            if col not in df.columns:
                return {"error": "El CSV debe tener columnas: ds, cpu, ram, trafico"}

    # 2. Forecast por métrica
    resultados     = {}
    fechas_colapso = {}
    meses_es = ["Enero","Febrero","Marzo","Abril","Mayo","Junio",
                "Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]

    for metrica in ["cpu", "ram", "trafico"]:
        filas, fechas_futuro, pred = forecast_metrica(df, metrica, periodos=12)
        resultados[metrica] = filas

        colapso_idx = next((i for i, v in enumerate(pred) if v >= 90), None)
        if colapso_idx is not None:
            f = fechas_futuro[colapso_idx]
            fechas_colapso[metrica] = f"{meses_es[f.month - 1]} {f.year}"
        else:
            fechas_colapso[metrica] = "No proyectado en 12 meses"

    # 3. Resumen ejecutivo via OpenAI REST API directa con httpx
    api_key    = os.environ.get("OPENAI_API_KEY")
    resumen_ia = ""
    if api_key:
        try:
            prompt = f"""Eres un arquitecto de redes senior presentando a la gerencia de una empresa de telecomunicaciones.
Basado en el análisis de forecasting (pronóstico) del router principal con proyección a 12 meses:
- CPU (Unidad Central de Procesamiento): saturación proyectada en {fechas_colapso.get('cpu')}
- RAM (Memoria de Acceso Aleatorio): saturación proyectada en {fechas_colapso.get('ram')}
- Tráfico de red: saturación proyectada en {fechas_colapso.get('trafico')}
Redacta un párrafo ejecutivo conciso (máximo 120 palabras) que describa el riesgo, mencione las fechas críticas, recomiende el reemplazo del equipo y use lenguaje comprensible para gerencia no técnica. Responde solo en español."""

            r = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={"model": "gpt-4o-mini",
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 200},
                timeout=20.0
            )
            resumen_ia = r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            resumen_ia = f"Error al conectar con OpenAI: {str(e)}"
    else:
        resumen_ia = "⚠️ Configura OPENAI_API_KEY en las variables de entorno para activar el resumen ejecutivo con IA."

    # 4. Histórico para el gráfico
    historico = df[["ds", "cpu", "ram", "trafico"]].copy()
    historico["ds"] = historico["ds"].astype(str)

    return {
        "historico":      historico.to_dict(orient="records"),
        "forecast":       resultados,
        "fechas_colapso": fechas_colapso,
        "resumen_ia":     resumen_ia
    }