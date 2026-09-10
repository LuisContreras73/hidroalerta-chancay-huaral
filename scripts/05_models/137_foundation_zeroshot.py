#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 137 — Modelos FUNDACIONALES de series de tiempo, evaluación ZERO-SHOT (gen6).

Pregunta del paper: ¿cuánto logra un modelo fundacional preentrenado, SIN ver jamás
esta cuenca, frente a los modelos locales entrenados (gen4) y las líneas base?

Modelos (corren en la RTX 4070 SUPER; VRAM pico < 1 GB):
  - Chronos-2  (Amazon):  paquete chronos-forecasting 2.3.1, checkpoint
    `amazon/chronos-2` (~120 M par., encoder multivariado, cuantiles nativos).
  - TimesFM 2.5 (Google): paquete timesfm 2.0.2 (torch), checkpoint
    `google/timesfm-2.5-200m-pytorch` (200 M par., decoder-only, cabeza continua
    de cuantiles). Layout de salida: [:,:,0]=media puntual, [:,:,1..9]=q0.1..q0.9.

Protocolo HONESTO (idéntico al de gen4 para comparabilidad directa):
  - Contexto: 512 días de la MISMA serie que ven los modelos locales
    (D7 sub_634 q_mm → m³/s; obs + reconstrucción GR4J donde no hay aforo).
  - Zero-shot puro: cero entrenamiento, cero tuning; el contexto termina el día
    de emisión t y se pronostica t+1..t+14.
  - Emisiones: todos los días t tales que algún objetivo t+L (L∈{1,3,7,14}) cae
    en la prueba 2024-01-01..2025-12-31.
  - Métricas SOLO contra aforo real (Regla 5), en los MISMOS días objetivo que
    forecast_multimodelo.csv (salidas 125→135) por lead.
  - CRPS = pinball medio sobre q∈{0.1,0.5,0.9} (misma fórmula que 114/125).

Salidas:
  outputs/ml_Q/137_chronos2_preds.csv · 137_timesfm25_preds.csv  (por día y lead)
  outputs/ml_Q/137_foundation_zeroshot.csv                        (métricas por lead)

Run en .venv313:
    python scripts/05_models/137_foundation_zeroshot.py
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
OUTML = ROOT / "outputs/ml_Q"
DASH = Path("D:/ANA Concurso/hidroalerta-dashboard/data")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("fm137")

Q90 = 40.89
Q_CONV = 86.4 / 3062.62
CTX = 512
H = 14
LEADS = [1, 3, 7, 14]
QLEV = [0.1, 0.5, 0.9]
TEST0, TEST1 = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")

# Checkpoints EN EL PROYECTO (materializados por scripts/09_ops/fetch_foundation_models.py,
# revisión clavada + MANIFEST SHA-256). Fallback al hub si aún no se han traído.
FM_DIR = ROOT / "models/foundation"
CHRONOS_CKPT = str(FM_DIR / "chronos-2") if (FM_DIR / "chronos-2/model.safetensors").exists()     else "amazon/chronos-2"
TIMESFM_CKPT = str(FM_DIR / "timesfm-2.5-200m") if (FM_DIR / "timesfm-2.5-200m/model.safetensors").exists()     else "google/timesfm-2.5-200m-pytorch"


def serie_contexto() -> pd.Series:
    d7 = pd.read_csv(ROOT / "data/model_ready/D7_multientity.csv", parse_dates=["date"])
    q = (d7[d7.entity_id == "sub_634"].set_index("date")["q_mm"] / Q_CONV).astype("float32")
    assert q.loc[:"2025-12-31"].isna().sum() == 0, "q_mm con NaN: el contexto exige serie continua"
    return q


def obs_real() -> pd.Series:
    obs = pd.read_csv(ROOT / "data/silver/snirh/S1_snirh_daily_q.csv",
                      parse_dates=["date"]).set_index("date")["q_santo_domingo_47e214d2"]
    return obs[~obs.index.duplicated(keep="last")]


def emisiones(q: pd.Series) -> pd.DatetimeIndex:
    dias = pd.date_range(TEST0 - pd.Timedelta(days=max(LEADS)), TEST1 - pd.Timedelta(days=1))
    return pd.DatetimeIndex([t for t in dias if t in q.index and t - pd.Timedelta(days=CTX - 1) >= q.index[0]])


def ventanas(q: pd.Series, ts: pd.DatetimeIndex) -> np.ndarray:
    v = np.stack([q.loc[:t].values[-CTX:] for t in ts]).astype("float32")
    assert v.shape == (len(ts), CTX)
    return v


def predecir_chronos2(v: np.ndarray) -> np.ndarray:
    """→ (n, H, 3) cuantiles q10/50/90."""
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(CHRONOS_CKPT, device_map="cuda",
                                               torch_dtype=torch.bfloat16)
    outs = []
    B = 128
    for i in range(0, len(v), B):
        x = torch.tensor(v[i:i + B]).unsqueeze(1)          # (b, 1, CTX)
        quants, _ = pipe.predict_quantiles(inputs=x, prediction_length=H,
                                           quantile_levels=QLEV)
        qq = torch.stack(quants) if isinstance(quants, list) else quants
        qq = qq.float().cpu().numpy()
        outs.append(qq.reshape(-1, H, len(QLEV)))
    del pipe
    torch.cuda.empty_cache()
    return np.concatenate(outs)


def predecir_timesfm(v: np.ndarray) -> np.ndarray:
    """→ (n, H, 3) cuantiles q10/50/90 (colummnas 1/5/9 del head de deciles)."""
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(TIMESFM_CKPT)
    m.compile(timesfm.ForecastConfig(max_context=CTX, max_horizon=64,
                                     normalize_inputs=True,
                                     use_continuous_quantile_head=True,
                                     fix_quantile_crossing=True))
    outs = []
    B = 64
    for i in range(0, len(v), B):
        pt, qt = m.forecast(horizon=H, inputs=list(v[i:i + B]))
        qt = np.asarray(qt)                                # (b, H, 10)
        outs.append(qt[:, :, [1, 5, 9]])                   # q0.1, q0.5, q0.9
    return np.concatenate(outs)


def metricas(o, p50, p10=None, p90=None):
    o, p = np.asarray(o, float), np.asarray(p50, float)
    nse = 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)
    r = np.corrcoef(o, p)[0, 1]
    alpha = p.std() / o.std()
    beta = p.mean() / o.mean()
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    mae = float(np.mean(np.abs(o - p)))
    crps = np.nan
    if p10 is not None:
        t = 0.0
        for qv, arr in zip(QLEV, [p10, p50, p90]):
            e = o - np.asarray(arr, float)
            t += np.mean(np.where(e >= 0, qv * e, (qv - 1) * e))
        crps = t / len(QLEV)
    oa, pa = o >= Q90, p >= Q90
    tp, fp, fn = np.sum(oa & pa), np.sum(~oa & pa), np.sum(oa & ~pa)
    csi = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    pod = tp / (tp + fn) if (tp + fn) else 0.0
    far = fp / (tp + fp) if (tp + fp) else 0.0
    return dict(NSE=round(nse, 3), KGE=round(kge, 3), MAE=round(mae, 2),
                CRPS=round(float(crps), 3), CSI=round(csi, 3),
                POD=round(pod, 3), FAR=round(far, 3))


def main():
    q = serie_contexto()
    obs = obs_real()
    ts = emisiones(q)
    log.info(f"{len(ts)} emisiones ({ts[0].date()} → {ts[-1].date()}), contexto {CTX} d")
    v = ventanas(q, ts)

    # días objetivo publicados, por lead (para evaluar EXACTAMENTE los mismos días)
    fc = pd.read_csv(DASH / "forecast_multimodelo.csv", parse_dates=["date"])
    dias_pub = {L: set(fc[(fc.model == "RA-TFT") & (fc.lead == L)]
                       .dropna(subset=["obs"])["date"]) for L in LEADS}

    preds = {}
    log.info("Chronos-2 (amazon/chronos-2, zero-shot)…")
    preds["Chronos-2"] = predecir_chronos2(v)
    log.info("TimesFM 2.5 (google/timesfm-2.5-200m-pytorch, zero-shot)…")
    preds["TimesFM-2.5"] = predecir_timesfm(v)

    filas_met, filas_pred = [], []
    for nombre, P in preds.items():
        for L in LEADS:
            tgt = ts + pd.Timedelta(days=L)
            o = obs.reindex(tgt).values
            en_pub = np.array([d in dias_pub[L] for d in tgt])
            m = np.isfinite(o) & en_pub
            p10, p50, p90 = P[:, L - 1, 0], P[:, L - 1, 1], P[:, L - 1, 2]
            filas_met.append(dict(model=nombre, lead=L, N=int(m.sum()),
                                  **metricas(o[m], p50[m], p10[m], p90[m])))
            for j in np.where(m)[0]:
                filas_pred.append(dict(date=tgt[j].date(), model=nombre, lead=L,
                                       obs=round(float(o[j]), 2),
                                       p10=round(float(p10[j]), 2),
                                       p50=round(float(p50[j]), 2),
                                       p90=round(float(p90[j]), 2)))

    met = pd.DataFrame(filas_met)
    met.to_csv(OUTML / "137_foundation_zeroshot.csv", index=False)
    pr = pd.DataFrame(filas_pred)
    for nombre, sub in pr.groupby("model"):
        f = OUTML / f"137_{nombre.lower().replace('-', '').replace('.', '')}_preds.csv"
        sub.to_csv(f, index=False)
    log.info("\n" + met.to_string(index=False))

    # referencia gen4 en los mismos leads (tabla curada del dashboard)
    ref = pd.read_csv(DASH / "metricas_modelos.csv")
    ref = ref[ref.lead.isin(LEADS)][["lead", "model", "NSE", "CRPS", "POD", "FAR"]]
    log.info("\nReferencia gen4 (dashboard):\n" + ref.to_string(index=False))
    log.info(f"Guardado: 137_foundation_zeroshot.csv + preds por modelo en {OUTML}")


if __name__ == "__main__":
    main()
