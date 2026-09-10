#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 140 — ¿Cuándo se rompen? Inferencia de TODOS los modelos hasta 2027-12-31.

Diseño (idéntico al experimento FIG2, ahora multi-modelo):
  - Contexto: serie q (D7 sub_634, obs+GR4J) hasta 2025-12-31 (2 048 días para
    los fundacionales — ~5,6 años de estacionalidad a la vista).
  - Se pronostican 730 días (2026-01-01 → 2027-12-31), MUY por fuera del dominio
    de diseño de todos (14 días): el objetivo es ver el MODO de ruptura.
  - Verdad parcial: aforo real ene–may 2026 (SNIRH) → NSE de esa ventana.
  - RA-TFT: NO se recomputa — su rollout autoregresivo está medido en FIG2
    (colapsa a la constante 28,8 m³/s; NSE ene–may 2026 = −1,29). Se dibuja esa
    referencia documentada.

Modos de ruptura que cuantifica:
  (1) NSE ene–may 2026 (la única ventana con verdad);
  (2) amplitud estacional relativa del año 2027 predicho:
      [media(feb–abr 2027) − media(jul–sep 2026)] / lo mismo en climatología —
      100 % = conserva el ciclo; ~0 % = colapsó a una constante.

Salidas: reports/diagnostico/FIG13_rollout_2027_todos.png
         outputs/ml_Q/140_rollout_2027.csv (trayectorias diarias por modelo)

Run en .venv313:
    python scripts/06_eval/140_rollout_2027_todos.py
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
OUTML = ROOT / "outputs/ml_Q"
FIGDIR = ROOT / "reports/diagnostico"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("fm140")

Q_CONV = 86.4 / 3062.62
CTX = 2048
H = 730
T0 = pd.Timestamp("2026-01-01")
FECHAS = pd.date_range(T0, periods=H, freq="D")
RATFT_CONST = 28.8            # FIG2: valor de colapso del rollout autoregresivo
RATFT_NSE26 = -1.29           # FIG2: NSE ene–may 2026 del rollout

COL = {"Chronos-2": "#7B2D8E", "TimesFM-2.5": "#B8860B", "RA-TFT (rollout)": "#0B6E8C",
       "Persistencia": "#8494A0", "Climatología": "#2E8B6F", "obs": "#0C1E2A"}


def series_base():
    d7 = pd.read_csv(ROOT / "data/model_ready/D7_multientity.csv", parse_dates=["date"])
    q = (d7[d7.entity_id == "sub_634"].set_index("date")["q_mm"] / Q_CONV).astype("float32")
    q = q.loc[:"2025-12-31"]
    obs = pd.read_csv(ROOT / "data/silver/snirh/S1_snirh_daily_q.csv",
                      parse_dates=["date"]).set_index("date")["q_santo_domingo_47e214d2"]
    obs = obs[~obs.index.duplicated(keep="last")].loc["2026-01-01":]
    clim_src = q.loc[:"2023-12-31"]                     # solo pre-test (Regla 3)
    doy = clim_src.groupby(clim_src.index.dayofyear).mean()
    clim = pd.Series([doy.get(d.dayofyear, np.nan) for d in FECHAS], index=FECHAS)
    return q, obs, clim


def rollout_chronos(ctx: np.ndarray) -> pd.DataFrame:
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(str(ROOT / "models/foundation/chronos-2"),
                                               device_map="cuda", torch_dtype=torch.bfloat16)
    x = torch.tensor(ctx).reshape(1, 1, -1)
    quants, _ = pipe.predict_quantiles(inputs=x, prediction_length=H,
                                       quantile_levels=[0.1, 0.5, 0.9])
    qq = (quants[0] if isinstance(quants, list) else quants)
    qq = qq.float().cpu().numpy()
    qq = qq[0] if qq.ndim == 3 else qq                   # (H, 3)
    del pipe; torch.cuda.empty_cache()
    return pd.DataFrame(qq, index=FECHAS, columns=["p10", "p50", "p90"])


def rollout_timesfm(ctx: np.ndarray) -> pd.DataFrame:
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        str(ROOT / "models/foundation/timesfm-2.5-200m"))
    m.compile(timesfm.ForecastConfig(max_context=CTX, max_horizon=768,
                                     normalize_inputs=True,
                                     use_continuous_quantile_head=True,
                                     fix_quantile_crossing=True))
    pt, qt = m.forecast(horizon=H, inputs=[ctx])
    qt = np.asarray(qt)[0, :H, :]                        # (H, 10)
    return pd.DataFrame(qt[:, [1, 5, 9]], index=FECHAS, columns=["p10", "p50", "p90"])


def nse(o, p):
    m = np.isfinite(o) & np.isfinite(p)
    o, p = o[m], p[m]
    return 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)


def amplitud_rel(p50: pd.Series, clim: pd.Series) -> float:
    """Amplitud estacional del segundo año predicho, relativa a la climatológica."""
    hum = slice("2027-02-01", "2027-04-30")
    sec = slice("2026-07-01", "2026-09-30")
    a_pred = p50.loc[hum].mean() - p50.loc[sec].mean()
    a_clim = clim.loc[hum].mean() - clim.loc[sec].mean()
    return float(100 * a_pred / a_clim)


def main():
    q, obs, clim = series_base()
    ctx = q.values[-CTX:]
    log.info(f"contexto: {CTX} d (hasta {q.index[-1].date()}) → pronóstico {H} d")

    tray = {"Climatología": pd.DataFrame({"p50": clim}),
            "Persistencia": pd.DataFrame({"p50": pd.Series(ctx[-1], index=FECHAS)}),
            "RA-TFT (rollout)": pd.DataFrame({"p50": pd.Series(RATFT_CONST, index=FECHAS)})}
    log.info("Chronos-2: 730 días en un solo pase (64 parches de 16)…")
    tray["Chronos-2"] = rollout_chronos(ctx)
    log.info("TimesFM 2.5: 730 días (autoregresivo por parches de 128)…")
    tray["TimesFM-2.5"] = rollout_timesfm(ctx)

    # ── métricas de ruptura ──────────────────────────────────────────────────
    ven = slice("2026-01-01", "2026-05-31")
    o26 = obs.loc[ven]
    filas = []
    for m, df in tray.items():
        p = df["p50"].reindex(o26.index).values
        n26 = RATFT_NSE26 if m == "RA-TFT (rollout)" else nse(o26.values, p)
        amp = amplitud_rel(df["p50"], clim)
        filas.append(dict(modelo=m, NSE_ene_may_2026=round(float(n26), 2),
                          amplitud_2027_pct=round(amp, 0)))
    res = pd.DataFrame(filas)
    log.info("\n" + res.to_string(index=False))

    largo = pd.concat([df.assign(model=m, date=df.index) for m, df in tray.items()])
    largo.to_csv(OUTML / "140_rollout_2027.csv", index=False)

    # ── FIG13 ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14.5, 9),
                             gridspec_kw={"height_ratios": [1.6, 1]})
    ax = axes[0]
    hist = q.loc["2025-07-01":]
    ax.plot(hist.index, hist.values, color="#6E7F8C", lw=1.2,
            label="serie histórica (contexto)")
    ax.plot(obs.index, obs.values, color=COL["obs"], lw=2.2, label="aforo real 2026")
    for m in ["Climatología", "Chronos-2", "TimesFM-2.5", "RA-TFT (rollout)",
              "Persistencia"]:
        df = tray[m]
        ls = ":" if m in ("Persistencia",) else ("--" if m == "Climatología" else "-")
        ax.plot(df.index, df["p50"], ls, color=COL[m], lw=2, label=m)
        if "p10" in df:
            ax.fill_between(df.index, df["p10"], df["p90"], color=COL[m], alpha=0.10)
    ax.axvline(T0, color="#33414C", lw=1, ls=":")
    ax.text(T0, ax.get_ylim()[1] * 0.97, " emisión (2026-01-01)", fontsize=9,
            va="top", color="#33414C")
    ax.set_ylabel("caudal (m³/s)"); ax.legend(fontsize=8.5, ncol=3, loc="upper right")
    ax.set_title("(a) Trayectorias hasta 2027 — el MODO de ruptura de cada familia")
    ax.grid(alpha=0.3)

    ax = axes[1]
    z = slice("2026-01-01", "2026-05-31")
    ax.plot(o26.index, o26.values, color=COL["obs"], lw=2.4, label="aforo real")
    for m in ["Climatología", "Chronos-2", "TimesFM-2.5", "RA-TFT (rollout)"]:
        df = tray[m]["p50"].loc[z]
        r = res[res.modelo == m]["NSE_ene_may_2026"].iloc[0]
        ls = "--" if m == "Climatología" else "-"
        ax.plot(df.index, df.values, ls, color=COL[m], lw=2,
                label=f"{m} (NSE {r:+.2f})")
    ax.set_ylabel("caudal (m³/s)")
    ax.set_title("(b) La única ventana con verdad: enero–mayo 2026")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle("FIG13 · Inferencia al 2027: todos fuera de su dominio — quién colapsa "
                 "a constante, quién degenera en climatología y quién conserva el ciclo",
                 fontweight="bold", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "FIG13_rollout_2027_todos.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"FIG13 + 140_rollout_2027.csv listos")


if __name__ == "__main__":
    main()
