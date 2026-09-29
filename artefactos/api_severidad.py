"""API de triage de severidad (MVP). Ejecutar: python api_severidad.py  ->  http://localhost:5000"""
import csv
import json
import time
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

import severidad_features as sf

DIR = Path(__file__).resolve().parent
MODELO = joblib.load(DIR / "modelo_champion.pkl")
LOG = DIR / "predicciones_log.csv"
app = Flask(__name__)

FORMULARIO = """<!doctype html><meta charset="utf-8"><title>Triage de severidad</title>
<style>body{font-family:system-ui;max-width:560px;margin:2rem auto;padding:0 1rem}label{display:block;margin:.5rem 0}
input,select{width:100%;padding:.35rem}button{margin-top:1rem;padding:.5rem 1rem}pre{background:#f4f4f4;padding:1rem}</style>
<h2>Primer aviso de siniestro</h2><form id="f">
<label>Alcaldía<select name="ALCALDIA">{alcaldias}</select></label>
<label>Tipo de accidente<select name="TIPACCID">{tipos}</select></label>
<label>Ubicación<select name="URBANA"><option>Intersección</option><option>No intersección</option></select></label>
<label>Día<select name="DIASEMANA">{dias}</select></label>
<label>Mes (1-12)<input name="MES" type="number" min="1" max="12" value="1"></label>
<label>Hora (0-23)<input name="ID_HORA" type="number" min="0" max="23" value="12"></label>
<label>Automóviles<input name="AUTOMOVIL" type="number" min="0" max="9" value="1"></label>
<label>Motocicletas<input name="MOTOCICLET" type="number" min="0" max="9" value="0"></label>
<label>Bicicletas<input name="BICICLETA" type="number" min="0" max="9" value="0"></label>
<label>Camiones de carga<input name="CAMION" type="number" min="0" max="9" value="0"></label>
<button>Calcular prioridad</button></form><pre id="r"></pre>
<script>document.getElementById('f').onsubmit=async e=>{e.preventDefault();
const d=Object.fromEntries(new FormData(e.target));const r=await fetch('/predecir',{method:'POST',
headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
document.getElementById('r').textContent=JSON.stringify(await r.json(),null,2)}</script>"""


def _opciones(valores):
    return "".join(f"<option>{v}</option>" for v in valores)


@app.get("/")
def formulario():
    return FORMULARIO.format(alcaldias=_opciones(sf.ALCALDIAS.values()), tipos=_opciones(sf.TIPOS_ACCIDENTE), dias=_opciones(sf.DIAS))


@app.get("/salud")
def salud():
    return jsonify({"estado": "ok", "modelo": MODELO.metadatos.get("modelo"),
                    "ventana": MODELO.metadatos.get("ventana_entrenamiento"), "entrenado": MODELO.metadatos.get("entrenado")})


@app.post("/predecir")
def predecir():
    t0 = time.perf_counter()
    entrada = request.get_json(silent=True)
    if entrada is None:
        return jsonify({"error": "El cuerpo debe ser JSON"}), 400
    datos, errores = sf.validar_reporte(entrada)
    if errores:
        return jsonify({"errores": errores}), 422
    salida = MODELO.predecir(datos).round(4)
    respuesta = [{"P_S0": r.P_S0, "P_S1": r.P_S1, "P_S2": r.P_S2, "score": r.SCORE,
                  "prioridad": r.PRIORIDAD, "clase_probable": r.CLASE_PROBABLE} for r in salida.itertuples()]
    ms = (time.perf_counter() - t0) * 1000
    nuevo = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["timestamp", "entrada", "salida", "ms"])
        for e, s in zip(datos.to_dict("records"), respuesta):
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), json.dumps(e, ensure_ascii=False), json.dumps(s), f"{ms:.1f}"])
    return jsonify(respuesta[0] if isinstance(entrada, dict) else respuesta)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
