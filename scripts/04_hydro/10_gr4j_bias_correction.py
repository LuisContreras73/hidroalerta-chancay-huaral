#!/usr/bin/env python3
"""
Script 10: Corrección de sesgo del GR4J con Q observado (Quantile Mapping mensual).

Problema: el GR4J reconstruct tiene sesgo sistemático grave:
  - Estiaje (Q<10): sobreestima +6.9 m³/s
  - Crecidas (Q>60): subestima -31.4 m³/s
  - Estacional: época húmeda GR4J/obs=0.6, estiaje GR4J/obs=2.0
  → El GR4J "aplana" la señal. Esto crea el salto visible vs Q obs.

Solución: Quantile Mapping mensual GR4J → Q obs (Piani et al. 2010, igual que
PISCOp vs SENAMHI y ERA5 vs PISCOt). Mapea la CDF del GR4J a la del Q obs por mes.

ANTI-LEAKAGE: el QM se calibra SOLO con Q obs 2020-09 a 2023-12 (período de
train+val del fine-tuning). El test 2024-2025 NUNCA entra en la calibración.

Salida:
  data/gold/G1_q_sim_gr4j_corrected.csv  — GR4J corregido 1981-2025
  outputs/figures/basin/G02_gr4j_correction.png
"""
import logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

ROOT = Path(__file__).parent.parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger("gr4j_bias")

GR4J_CSV = ROOT/"data/gold/G1_q_sim_gr4j.csv"
QOBS_CSV = ROOT/"data/silver/snirh/S1_snirh_daily_q.csv"
OUT_CSV  = ROOT/"data/gold/G1_q_sim_gr4j_corrected.csv"
FIG      = ROOT/"outputs/figures/basin/G02_gr4j_correction.png"
FIG.parent.mkdir(parents=True, exist_ok=True)

CAL_END = "2023-12-31"   # QM calibrado solo hasta aquí (anti-leakage con test 2024+)
N_Q = 100                 # cuantiles


def fit_qm_monthly(sim_cal, obs_cal, n_q=N_Q):
    """QM mensual: para cada mes, mapea CDF(GR4J) → CDF(obs)."""
    qv = np.linspace(0, 1, n_q+1)
    params = {}
    for m in range(1, 13):
        s = sim_cal[sim_cal.index.month == m].dropna()
        o = obs_cal[obs_cal.index.month == m].dropna()
        if len(s) < 20 or len(o) < 20:
            params[m] = None
            continue
        params[m] = (np.quantile(s.values, qv), np.quantile(o.values, qv))
    return params


def apply_qm_monthly(sim_full, params):
    corr = sim_full.copy()
    for m in range(1, 13):
        p = params.get(m)
        mask = sim_full.index.month == m
        if p is None:
            continue
        sq, oq = p
        f = interp1d(sq, oq, kind="linear", bounds_error=False,
                     fill_value=(oq[0], oq[-1]))
        corr.loc[mask] = np.maximum(f(sim_full[mask].values), 0.0)
    return corr


def metrics(o, p):
    mask = np.isfinite(o) & np.isfinite(p)
    o, p = o[mask], p[mask]
    nse = 1 - np.sum((o-p)**2)/(np.sum((o-o.mean())**2)+1e-12)
    r = np.corrcoef(o,p)[0,1]
    a, b = p.std()/(o.std()+1e-12), p.mean()/(o.mean()+1e-12)
    kge = 1 - np.sqrt((r-1)**2+(a-1)**2+(b-1)**2)
    pbias = (p.sum()-o.sum())/(o.sum()+1e-12)*100
    return nse, kge, pbias


def main():
    log.info("Script 10: corrección de sesgo GR4J (QM mensual)")
    gr4j_df = pd.read_csv(GR4J_CSV, index_col=0, parse_dates=True)
    gr4j = gr4j_df["q_sim_m3s"]
    qobs = pd.read_csv(QOBS_CSV, index_col=0, parse_dates=True)["q_santo_domingo_47e214d2"].dropna()

    # Calibrar QM solo con período <= CAL_END (anti-leakage)
    common = gr4j.index.intersection(qobs.index)
    cal = common[common <= CAL_END]
    val = common[common > CAL_END]
    log.info(f"Calibración QM: {len(cal)} días (≤{CAL_END}) | validación: {len(val)} días (>{CAL_END})")

    params = fit_qm_monthly(gr4j.loc[cal], qobs.loc[cal])
    n_ok = sum(1 for v in params.values() if v is not None)
    log.info(f"QM ajustado en {n_ok}/12 meses")

    # Aplicar a todo el GR4J 1981-2025
    gr4j_corr = apply_qm_monthly(gr4j, params)

    # Métricas antes/después
    log.info("\n=== MÉTRICAS GR4J vs Q obs ===")
    for label, idx in [("Calibración ≤2023", cal), ("Validación 2024+", val)]:
        if len(idx) < 10:
            continue
        o = qobs.loc[idx].values
        n0,k0,p0 = metrics(o, gr4j.loc[idx].values)
        n1,k1,p1 = metrics(o, gr4j_corr.loc[idx].values)
        log.info(f"{label}:")
        log.info(f"   ANTES:  NSE={n0:.3f} KGE={k0:.3f} PBIAS={p0:.1f}%")
        log.info(f"   DESPUÉS:NSE={n1:.3f} KGE={k1:.3f} PBIAS={p1:.1f}%")

    # Guardar
    out = gr4j_df.copy()
    out["q_sim_m3s_corrected"] = gr4j_corr.values
    out["q_sim_mm_corrected"] = gr4j_corr.values * (86.4/3062.62)
    out.to_csv(OUT_CSV)
    log.info(f"\nGuardado: {OUT_CSV.name}")

    # ── Figura comparativa ──────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Q90 = 40.89
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Corrección de Sesgo GR4J por Quantile Mapping mensual (vs Q observado)\n"
                 "Antes: GR4J aplana la señal (sobreestima estiaje, subestima picos)",
                 fontsize=12, fontweight="bold")

    # Panel 1: serie temporal período común
    ax = axes[0][0]
    cm = common[(common>="2021-01-01")&(common<="2023-12-31")]
    ax.plot(cm, qobs.loc[cm], color="#c0392b", lw=1.3, label="Q observado")
    ax.plot(cm, gr4j.loc[cm], color="#7f8c8d", lw=1.0, ls="--", label="GR4J original")
    ax.plot(cm, gr4j_corr.loc[cm], color="#27ae60", lw=1.0, label="GR4J corregido")
    ax.axhline(Q90, color="orange", lw=0.8, ls=":")
    ax.set_ylabel("Q (m³/s)"); ax.set_title("Serie 2021-2023 (calibración)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0)

    # Panel 2: scatter antes
    ax2 = axes[0][1]
    o_all = qobs.loc[cal].values
    ax2.scatter(o_all, gr4j.loc[cal].values, s=8, alpha=0.3, color="#7f8c8d", label="GR4J original")
    ax2.scatter(o_all, gr4j_corr.loc[cal].values, s=8, alpha=0.3, color="#27ae60", label="GR4J corregido")
    lim = max(o_all.max(), gr4j.loc[cal].max())*1.05
    ax2.plot([0,lim],[0,lim],"k--",lw=0.8)
    ax2.set_xlabel("Q observado"); ax2.set_ylabel("GR4J"); ax2.set_title("Scatter (calibración)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # Panel 3: climatología mensual
    ax3 = axes[1][0]
    meses=["E","F","M","A","M","J","J","A","S","O","N","D"]
    o_clim = qobs.loc[cal].groupby(qobs.loc[cal].index.month).mean()
    g_clim = gr4j.loc[cal].groupby(gr4j.loc[cal].index.month).mean()
    gc_clim = gr4j_corr.loc[cal].groupby(gr4j_corr.loc[cal].index.month).mean()
    ax3.plot(o_clim.index, o_clim.values, color="#c0392b", lw=2, marker="o", label="Q obs")
    ax3.plot(g_clim.index, g_clim.values, color="#7f8c8d", lw=1.5, marker="s", ls="--", label="GR4J orig")
    ax3.plot(gc_clim.index, gc_clim.values, color="#27ae60", lw=1.5, marker="^", label="GR4J corr")
    ax3.set_xticks(range(1,13)); ax3.set_xticklabels(meses)
    ax3.set_xlabel("Mes"); ax3.set_ylabel("Q medio (m³/s)")
    ax3.set_title("Climatología mensual (corrige sesgo estacional)")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # Panel 4: CDF / FDC
    ax4 = axes[1][1]
    for serie, c, lbl in [(qobs.loc[cal],"#c0392b","Q obs"),
                          (gr4j.loc[cal],"#7f8c8d","GR4J orig"),
                          (gr4j_corr.loc[cal],"#27ae60","GR4J corr")]:
        s = np.sort(serie.dropna().values)[::-1]
        exc = np.arange(1,len(s)+1)/(len(s)+1)*100
        ax4.plot(exc, s, color=c, lw=1.5, label=lbl)
    ax4.axhline(Q90, color="orange", lw=0.8, ls=":")
    ax4.set_xlabel("Excedencia (%)"); ax4.set_ylabel("Q (m³/s)")
    ax4.set_title("Curva de duración (FDC)"); ax4.set_yscale("log")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
    log.info(f"Figura: {FIG.name}")


if __name__ == "__main__":
    main()
