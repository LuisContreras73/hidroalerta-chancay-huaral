#!/usr/bin/env python3
"""
Script 04_hydro/09_calibracion_gr4j.py — Calibración GR4J con DSST (Klemeš 1986).

Experimentos:
  E1  Cal_seco  → Val_húmedo  — DSST: robustez climática
  E2  Cal_húmedo → Val_seco   — DSST: robustez climática inversa
  E3  SST cronológico          — Cal 60% inicial / Val 40% final (Q obs)
  E4  Calibración final        — todos los días Q obs disponibles

Gate obligatorio (D055): NSE_val E1 ≥ 0.60 AND NSE_val E2 ≥ 0.60
→ Aprueba uso de reconstrucción GR4J 1981–2020 como blanco de entrenamiento ML.

Función objetivo multi-criterio (D052):
  J = 0.4×NSE + 0.3×KGE + 0.2×NSE_log − 0.1×|PBIAS_Q90|

Períodos DSST:
  Dry (La Niña)  : 2020-09-01 → 2022-08-31
  Wet (El Niño)  : 2022-09-01 → fin datos Q
  Warm-up (opt.) : 2017-09-01 → 2020-08-31  (3 años; suficiente para GR4J)

Forzantes:
  PR  : data/bronze/B2_pisco_v3_basin_mean.csv       (PISCOp v3.0, D050)
  PET : data/bronze/B2_pet_era5pm_1981_2025.csv       (hscal 1981-2020 + ERA5 pev cal 2021-2025; script 08f)

Parámetros GR4J y rangos de búsqueda (D052):
  X1 [mm]  : almacén producción         [100, 2000]
  X2 [mm]  : intercambio subterráneo    [−5, 3]
  X3 [mm]  : almacén enrutamiento       [1, 1000]
  X4 [días]: base UH                    [0.5, 10]

Outputs:
  configs/gr4j_params.yaml              — parámetros E4 + métricas DSST
  data/gold/G1_q_sim_gr4j.csv           — reconstrucción 1981-2025
  data/gold/G1_gr4j_states_d6.csv       — estados (q_gr4j_mm, S_norm, R_norm)
  outputs/figures/basin/G01_gr4j_dsst.png

Referencias:
  Perrin et al. (2003) doi:10.1016/S0022-1694(03)00225-7   GR4J
  Klemeš (1986) doi:10.1080/02626668609491024              DSST
  Gupta et al. (2009) doi:10.1029/2008WR007819             KGE
  Storn & Price (1997) doi:10.1023/A:1008202821328         DE
"""
import datetime
import logging
import warnings
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import yaml

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 9})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("gr4j_dsst")

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent.parent
PR_FILE    = ROOT / "data/bronze/B2_pisco_v3_basin_mean.csv"
PET_FILE   = ROOT / "data/bronze/B2_pet_era5pm_1981_2025.csv"
Q_FILE     = ROOT / "data/silver/snirh/S1_snirh_daily_q.csv"
OUT_PARAMS = ROOT / "configs/gr4j_params.yaml"
OUT_Q_SIM  = ROOT / "data/gold/G1_q_sim_gr4j.csv"
OUT_STATES = ROOT / "data/gold/G1_gr4j_states_d6.csv"
FIG_DIR    = ROOT / "outputs/figures/basin"
OUT_FIG    = FIG_DIR / "G01_gr4j_dsst.png"

# ── Constantes ────────────────────────────────────────────────────────────────
AREA_KM2 = 3062.62
Q_COL    = "q_santo_domingo_47e214d2"

DRY_START = pd.Timestamp("2020-09-01")   # La Niña
DRY_END   = pd.Timestamp("2022-08-31")
WET_START = pd.Timestamp("2022-09-01")   # El Niño 2023-24
OPT_START = pd.Timestamp("2017-09-01")   # inicio warm-up optimización

BOUNDS = [
    (100.0, 2000.0),   # X1 [mm]
    (-5.0,    3.0),    # X2 [mm]
    (1.0,  1000.0),    # X3 [mm]
    (0.5,    10.0),    # X4 [días]
]

# Criterios de aceptación DSST (D055)
NSE_THR_DSST = 0.60
NSE_THR_SST  = 0.65
NSE_THR_CAL  = 0.70
KGE_THR_DSST = 0.55
PBIAS_THR    = 20.0


# ══════════════════════════════════════════════════════════════════════════════
# GR4J CORE (Perrin et al., 2003)
# ══════════════════════════════════════════════════════════════════════════════
def _build_uh(X4: float) -> tuple[np.ndarray, np.ndarray]:
    """Hidrogramas unitarios UH1 y UH2 (Perrin 2003, Apéndice A)."""
    n1 = int(np.ceil(X4))
    n2 = int(np.ceil(2 * X4))

    sh1 = np.zeros(n1 + 1)
    for j in range(1, n1 + 1):
        t = float(j)
        sh1[j] = (t / X4) ** 2.5 if t <= X4 else 1.0
    uh1 = np.diff(sh1)

    sh2 = np.zeros(n2 + 1)
    for j in range(1, n2 + 1):
        t = float(j)
        if t <= X4:
            sh2[j] = 0.5 * (t / X4) ** 2.5
        elif t <= 2 * X4:
            sh2[j] = 1.0 - 0.5 * (2.0 - t / X4) ** 2.5
        else:
            sh2[j] = 1.0
    uh2 = np.diff(sh2)
    return uh1, uh2


def gr4j(
    P: np.ndarray,
    E: np.ndarray,
    X1: float,
    X2: float,
    X3: float,
    X4: float,
    S0: float | None = None,
    R0: float | None = None,
    return_states: bool = False,
):
    """
    GR4J daily rainfall-runoff model.

    Parameters  P, E [mm/d]; X1..X4 parámetros.
    Returns     (Q, S_final, R_final) [mm/d, mm, mm]
                + (S_ts, R_ts) si return_states=True
    """
    n        = len(P)
    uh1, uh2 = _build_uh(X4)
    n1, n2   = len(uh1), len(uh2)
    S        = 0.5 * X1 if S0 is None else float(S0)
    R        = 0.5 * X3 if R0 is None else float(R0)
    buf1     = np.zeros(n1)
    buf2     = np.zeros(n2)
    Q        = np.zeros(n)
    S_ts     = np.zeros(n) if return_states else None
    R_ts     = np.zeros(n) if return_states else None

    for t in range(n):
        Pt, Et = float(P[t]), float(E[t])
        if Pt >= Et:
            Pn, En = Pt - Et, 0.0
        else:
            Pn, En = 0.0, Et - Pt

        if Pn > 0.0:
            tmp = np.tanh(Pn / X1)
            sv  = S / X1
            Ps  = X1 * (1.0 - sv * sv) * tmp / (1.0 + sv * tmp)
        else:
            Ps = 0.0

        if En > 0.0:
            tmp = np.tanh(En / X1)
            sv  = S / X1
            Es  = S * (2.0 - sv) * tmp / (1.0 + (1.0 - sv) * tmp)
        else:
            Es = 0.0

        S    = max(0.0, S - Es + Ps)
        Perc = S * (1.0 - (1.0 + (4.0 * S / (9.0 * X1)) ** 4) ** (-0.25))
        S   -= Perc
        PR   = Perc + Pn - Ps

        buf1[1:] = buf1[:-1]; buf1[0] = 0.9 * PR
        Q9       = float(np.dot(buf1, uh1))
        buf2[1:] = buf2[:-1]; buf2[0] = 0.1 * PR
        Q1       = float(np.dot(buf2, uh2))

        F  = X2 * (max(R, 0.0) / X3) ** 3.5
        R  = max(0.0, R + Q9 + F)
        rv = R / X3
        Qr = R * (1.0 - (1.0 + rv ** 4) ** (-0.25))
        R -= Qr

        Q[t] = Qr + max(0.0, Q1 + F)
        if return_states:
            S_ts[t] = S
            R_ts[t] = R

    if return_states:
        return Q, S, R, S_ts, R_ts
    return Q, S, R


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS Y FUNCIÓN OBJETIVO
# ══════════════════════════════════════════════════════════════════════════════
def _metrics(obs: np.ndarray, sim: np.ndarray) -> dict:
    mask = ~(np.isnan(obs) | np.isnan(sim))
    o, s = obs[mask], sim[mask]
    if len(o) < 10:
        return {k: float("nan") for k in
                ["KGE", "NSE", "logNSE", "PBIAS", "PBIAS_Q90", "r", "RMSE", "N"]}
    r      = float(np.corrcoef(o, s)[0, 1])
    alpha  = s.std() / (o.std() + 1e-9)
    beta   = s.mean() / (o.mean() + 1e-9)
    kge    = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    nse    = 1.0 - np.sum((o - s) ** 2) / (np.sum((o - o.mean()) ** 2) + 1e-9)
    wet    = (o > 0.1) & (s > 0.0)
    lnse   = (1.0 - np.sum((np.log(o[wet] + 1) - np.log(s[wet] + 1)) ** 2)
              / (np.sum((np.log(o[wet] + 1) - np.log(o[wet] + 1).mean()) ** 2) + 1e-9)
              ) if wet.sum() > 5 else float("nan")
    pbias  = (s.sum() - o.sum()) / (o.sum() + 1e-9) * 100.0
    q90    = np.nanpercentile(o, 90)
    pk     = o > q90
    pbias_q90 = ((s[pk].sum() - o[pk].sum()) / (o[pk].sum() + 1e-9) * 100.0
                 ) if pk.sum() > 5 else float("nan")
    rmse   = float(np.sqrt(np.mean((o - s) ** 2)))
    return dict(
        KGE=round(float(kge), 3),
        NSE=round(float(nse), 3),
        logNSE=round(float(lnse), 3) if not np.isnan(lnse) else float("nan"),
        PBIAS=round(float(pbias), 1),
        PBIAS_Q90=round(float(pbias_q90), 1) if not np.isnan(pbias_q90) else float("nan"),
        r=round(float(r), 3),
        RMSE=round(float(rmse), 3),
        N=int(len(o)),
    )


def _multi_obj(obs: np.ndarray, sim: np.ndarray) -> float:
    """J = 0.4×NSE + 0.3×KGE + 0.2×NSE_log − 0.1×|PBIAS_Q90|/100  (D052)."""
    m = _metrics(obs, sim)
    if np.isnan(m["NSE"]) or np.isnan(m["KGE"]):
        return -2.0
    lnse      = m["logNSE"]    if not np.isnan(m["logNSE"])    else m["NSE"]
    pbias_q90 = m["PBIAS_Q90"] if not np.isnan(m["PBIAS_Q90"]) else 0.0
    return 0.4 * m["NSE"] + 0.3 * m["KGE"] + 0.2 * lnse - 0.1 * abs(pbias_q90) / 100.0


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTO DSST
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment(
    label: str,
    P: np.ndarray,
    E: np.ndarray,
    Q_obs: np.ndarray,
    cal_mask: np.ndarray,
    val_mask: np.ndarray | None,
    bounds: list,
    seed: int = 42,
    maxiter: int = 300,
) -> dict:
    """Calibra GR4J minimizando −J en cal_mask; evalúa en val_mask."""
    n_cal = int(cal_mask.sum())
    n_val = int(val_mask.sum()) if val_mask is not None else 0
    log.info(f"\n{'─' * 60}")
    log.info(f"  {label}")
    log.info(f"  Cal días válidos: {n_cal}  |  Val días: {n_val}")

    call_count = [0]

    def obj(params):
        call_count[0] += 1
        Q_sim, _, _ = gr4j(P, E, *params)
        return -_multi_obj(Q_obs[cal_mask], Q_sim[cal_mask])

    result = opt.differential_evolution(
        obj, bounds, seed=seed, maxiter=maxiter,
        popsize=12, tol=1e-6, mutation=(0.5, 1.5),
        recombination=0.7, workers=1, disp=False, polish=True,
    )
    params = result.x
    X1, X2, X3, X4 = params
    log.info(f"  Convergencia: {result.message} | Evals: {call_count[0]}")
    log.info(f"  X1={X1:.1f}  X2={X2:.4f}  X3={X3:.1f}  X4={X4:.3f}")

    Q_sim_all, _, _ = gr4j(P, E, *params)
    m_cal = _metrics(Q_obs[cal_mask], Q_sim_all[cal_mask])
    m_val = _metrics(Q_obs[val_mask], Q_sim_all[val_mask]) if val_mask is not None else {}

    log.info(f"  Cal: NSE={m_cal.get('NSE','—')}  KGE={m_cal.get('KGE','—')}  "
             f"PBIAS={m_cal.get('PBIAS','—')}%  N={m_cal.get('N','—')}")
    if val_mask is not None:
        log.info(f"  Val: NSE={m_val.get('NSE','—')}  KGE={m_val.get('KGE','—')}  "
                 f"PBIAS={m_val.get('PBIAS','—')}%  N={m_val.get('N','—')}")

    return {
        "label"      : label,
        "params"     : params,
        "metrics_cal": m_cal,
        "metrics_val": m_val,
        "Q_sim"      : Q_sim_all,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA DSST
# ══════════════════════════════════════════════════════════════════════════════
def make_dsst_figure(
    results: list,
    Q_obs: np.ndarray,
    opt_idx: pd.DatetimeIndex,
    dry_mask: np.ndarray,
    wet_mask: np.ndarray,
) -> None:
    q_obs_m3s = Q_obs * AREA_KM2 / 86.4
    EXP_COLOR = {"E1": "#2980b9", "E2": "#e74c3c", "E3": "#27ae60"}
    TITLES = {
        "E1": "(a) E1 — Cal_seco → Val_húmedo (DSST)",
        "E2": "(b) E2 — Cal_húmedo → Val_seco (DSST)",
        "E3": "(c) E3 — SST cronológico (Cal 60% / Val 40%)",
    }

    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.58, wspace=0.35)
    POSITIONS = {"E1": gs[0, :], "E2": gs[1, :], "E3": gs[2, 0]}

    def _shade(ax):
        if dry_mask.any():
            ax.axvspan(opt_idx[dry_mask][0], opt_idx[dry_mask][-1],
                       alpha=0.10, color="#2980b9", label="Seco (La Niña)")
        if wet_mask.any():
            ax.axvspan(opt_idx[wet_mask][0], opt_idx[wet_mask][-1],
                       alpha=0.10, color="#e74c3c", label="Húmedo (El Niño)")

    for exp_id, pos in POSITIONS.items():
        r = next((x for x in results if x["label"].startswith(exp_id)), None)
        if r is None:
            continue
        ax = fig.add_subplot(pos)
        q_sim_m3s = r["Q_sim"] * AREA_KM2 / 86.4
        ax.plot(opt_idx, q_obs_m3s, color="#444", lw=1.0, alpha=0.85, label="Q obs")
        ax.plot(opt_idx, q_sim_m3s, color=EXP_COLOR[exp_id], lw=0.9,
                alpha=0.85, label=f"Q sim ({exp_id})")
        _shade(ax)
        m_c, m_v = r["metrics_cal"], r["metrics_val"]
        txt = (f"Cal  NSE={m_c.get('NSE','—'):.3f}  KGE={m_c.get('KGE','—'):.3f}\n"
               f"Val  NSE={m_v.get('NSE','—'):.3f}  KGE={m_v.get('KGE','—'):.3f}"
               if m_v else f"Cal  NSE={m_c.get('NSE','—'):.3f}  KGE={m_c.get('KGE','—'):.3f}")
        ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", fontsize=8,
                family="monospace",
                bbox=dict(fc="white", ec="gray", alpha=0.9, pad=3, lw=0.8))
        ax.set_title(TITLES[exp_id], fontweight="bold", fontsize=10)
        ax.set_ylabel("Caudal (m³/s)", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3, lw=0.7)
        ax.legend(fontsize=8, loc="upper right", ncol=2)

    # Panel (d): tabla criterios DSST
    ax_t = fig.add_subplot(gs[2, 1])
    ax_t.axis("off")
    THR_NSE = {"E1": NSE_THR_DSST, "E2": NSE_THR_DSST, "E3": NSE_THR_SST}
    rows = []
    for r in results[:3]:
        eid   = r["label"][:2]
        m_c   = r["metrics_cal"]
        m_v   = r["metrics_val"]
        nse_c = m_c.get("NSE", float("nan"))
        nse_v = m_v.get("NSE", float("nan"))
        kge_v = m_v.get("KGE", float("nan"))
        thr   = THR_NSE.get(eid, 0.60)
        gate  = "PASS" if (not np.isnan(nse_v) and nse_v >= thr) else "FAIL"
        rows.append([
            eid,
            f"{nse_c:.3f}",
            f"{nse_v:.3f}" if not np.isnan(nse_v) else "—",
            f"{kge_v:.3f}" if not np.isnan(kge_v) else "—",
            f"≥{thr}",
            gate,
        ])
    headers = ["Exp", "NSE cal", "NSE val", "KGE val", "Umbral", "Gate"]
    tbl = ax_t.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.15, 1.8)
    for (ri, ci), cell in tbl.get_celld().items():
        if ri == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif ri <= len(rows):
            gate_val = rows[ri - 1][-1]
            cell.set_facecolor("#d5f5e3" if gate_val == "PASS" else "#fadbd8")
    ax_t.set_title("(d) Criterios de aceptación DSST (D055)",
                   fontweight="bold", fontsize=10)

    fig.suptitle(
        "Calibración GR4J — Differential Split-Sample Test (Klemeš 1986)\n"
        "Cuenca Chancay-Huaral  |  PR: PISCOp v3.0  |  "
        f"Área={AREA_KM2:.0f} km²  |  DE seed=42 maxiter=300",
        fontsize=11, fontweight="bold",
    )
    plt.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
    plt.close()
    log.info(f"Figura guardada: {OUT_FIG.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    log.info("=" * 70)
    log.info("Script 04_hydro/09 — Calibración GR4J con DSST (Klemeš 1986)")
    log.info("=" * 70)

    for f in (PR_FILE, PET_FILE, Q_FILE):
        if not f.exists():
            raise FileNotFoundError(f"Archivo fuente no encontrado: {f}")
    (ROOT / "data/gold").mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Forzantes PR ──────────────────────────────────────────────────────
    pr = pd.read_csv(PR_FILE, index_col=0, parse_dates=True).squeeze("columns")
    pr.index = pr.index.normalize()
    pr.name  = "pr"
    log.info(f"PR (PISCOp v3): {pr.index.min().date()} → {pr.index.max().date()}  N={len(pr)}")

    # ── 2. Forzantes PET — tres fases: hscal 1981-2020 + ERA5 pev cal 2021-2025 ─
    pet_df = pd.read_csv(PET_FILE, index_col=0, parse_dates=True)
    pet_df.index = pet_df.index.normalize()
    pet_full_raw = pet_df["pet"]          # column "pet" = three-phase product
    full_idx = pd.date_range("1981-01-01", "2025-12-31", freq="D")
    pet_full  = pet_full_raw.reindex(full_idx)
    nan_count = int(pet_full.isna().sum())
    log.info(f"PET (08f): {pet_full_raw.index.min().date()} → {pet_full_raw.index.max().date()}"
             f"  N={len(pet_full_raw)}  NaN={nan_count}")
    if nan_count > 0:
        log.warning(f"PET tiene {nan_count} NaN — completando con climatología de emergencia")
        pet_clim = pet_full.groupby(pet_full.index.month).mean()
        gap_mask = pet_full.isna()
        pet_full.loc[gap_mask] = pet_full.index[gap_mask].month.map(pet_clim)

    # ── 3. Q obs ─────────────────────────────────────────────────────────────
    df_q = pd.read_csv(Q_FILE, index_col=0, parse_dates=True)
    df_q.index = df_q.index.normalize()
    q_obs_m3s = df_q[Q_COL].reindex(full_idx)
    q_obs_mm  = q_obs_m3s * 86.4 / AREA_KM2
    n_valid   = int(q_obs_mm.notna().sum())
    log.info(f"Q obs {Q_COL}: {df_q.index.min().date()} → {df_q.index.max().date()}")
    log.info(f"  Válidos: {n_valid} / {len(q_obs_mm)} días")

    # ── 4. Período de optimización ────────────────────────────────────────────
    q_end   = min(df_q.index.max(), pd.Timestamp("2025-12-31"))
    opt_idx = pd.date_range(OPT_START, q_end, freq="D")

    P_opt   = pr.reindex(opt_idx).fillna(0.0).values
    E_opt   = pet_full.reindex(opt_idx).values
    Q_opt   = q_obs_mm.reindex(opt_idx).values

    # Máscaras booleanas (opt_idx comparisons already return np.ndarray)
    dry_mask = (opt_idx >= DRY_START) & (opt_idx <= DRY_END)
    wet_all  = opt_idx >= WET_START                             # todos los días húmedos
    dry_cal  = dry_mask & ~np.isnan(Q_opt)                     # días secos con Q válido
    wet_cal  = wet_all  & ~np.isnan(Q_opt)                     # días húmedos con Q válido

    # SST: corte cronológico 60/40 sobre días con Q válido
    valid_dates   = opt_idx[~np.isnan(Q_opt)]
    sst_split     = valid_dates[int(len(valid_dates) * 0.60)]
    sst_cal_mask  = (~np.isnan(Q_opt)) & (opt_idx < sst_split)
    sst_val_mask  = (~np.isnan(Q_opt)) & (opt_idx >= sst_split)

    all_q_mask = ~np.isnan(Q_opt)

    log.info(f"\nPeríodos (opt_idx: {OPT_START.date()} → {q_end.date()}):")
    log.info(f"  Dry  : {DRY_START.date()} → {DRY_END.date()}  "
             f"({dry_cal.sum()} días válidos)")
    log.info(f"  Wet  : {WET_START.date()} → {q_end.date()}  "
             f"({wet_cal.sum()} días válidos)")
    log.info(f"  SST split: {sst_split.date()}  "
             f"Cal={sst_cal_mask.sum()}d  Val={sst_val_mask.sum()}d")
    log.info(f"  Total Q válidos en opt: {all_q_mask.sum()}")

    # ── 5. Experimentos E1 – E4 ───────────────────────────────────────────────
    results = [
        run_experiment(
            "E1 Cal_seco → Val_húmedo (DSST)",
            P_opt, E_opt, Q_opt, dry_cal, wet_all, BOUNDS,
        ),
        run_experiment(
            "E2 Cal_húmedo → Val_seco (DSST)",
            P_opt, E_opt, Q_opt, wet_cal, dry_mask, BOUNDS,
        ),
        run_experiment(
            "E3 SST cronológico (60/40)",
            P_opt, E_opt, Q_opt, sst_cal_mask, sst_val_mask, BOUNDS,
        ),
        run_experiment(
            "E4 Final — todos Q obs",
            P_opt, E_opt, Q_opt, all_q_mask, None, BOUNDS,
        ),
    ]

    # ── 6. Gate de aceptación DSST (D055) ────────────────────────────────────
    nse_e1 = results[0]["metrics_val"].get("NSE", float("nan"))
    nse_e2 = results[1]["metrics_val"].get("NSE", float("nan"))
    gate_pass = (
        not np.isnan(nse_e1) and nse_e1 >= NSE_THR_DSST and
        not np.isnan(nse_e2) and nse_e2 >= NSE_THR_DSST
    )
    log.info("\n" + "─" * 60)
    log.info("GATE DSST (D055):")
    log.info(f"  E1 NSE_val = {nse_e1:.3f}  "
             f"({'≥' if nse_e1 >= NSE_THR_DSST else '<'} {NSE_THR_DSST})")
    log.info(f"  E2 NSE_val = {nse_e2:.3f}  "
             f"({'≥' if nse_e2 >= NSE_THR_DSST else '<'} {NSE_THR_DSST})")
    log.info(f"  Resultado  : {'✓ PASS — GR4J aprobado como blanco ML' if gate_pass else '✗ FAIL — revisar calibración'}")
    log.info("─" * 60)
    if not gate_pass:
        log.warning("Gate DSST no aprobado. Parámetros E4 guardados de todos modos.")

    # ── 7. Reconstrucción histórica 1981-2025 con parámetros E4 ───────────────
    log.info("\nReconstrucción histórica 1981-2025 con parámetros E4 ...")
    X1, X2, X3, X4 = results[3]["params"]
    hist_idx   = pd.date_range("1981-01-01", "2025-12-31", freq="D")
    P_hist     = pr.reindex(hist_idx).fillna(0.0).values
    E_hist     = pet_full.reindex(hist_idx).values
    Q_hist_mm, _, _, S_ts, R_ts = gr4j(
        P_hist, E_hist, X1, X2, X3, X4, return_states=True
    )
    Q_hist_m3s = Q_hist_mm * AREA_KM2 / 86.4

    df_out = pd.DataFrame({
        "pr_mm_day"   : P_hist,
        "pet_mm_day"  : E_hist,
        "q_sim_mm_day": Q_hist_mm,
        "q_sim_m3s"   : Q_hist_m3s,
        "S_mm"        : S_ts,
        "R_mm"        : R_ts,
        "S_norm"      : np.clip(S_ts / (X1 + 1e-9), 0.0, 1.0),
        "R_norm"      : np.clip(R_ts / (X3 + 1e-9), 0.0, 1.0),
    }, index=hist_idx)
    df_out.index.name = "date"
    df_out.to_csv(OUT_Q_SIM)
    log.info(f"  {OUT_Q_SIM.name}  (N={len(df_out)}, Q_mean={Q_hist_m3s.mean():.2f} m³/s)")

    df_states = df_out[["q_sim_mm_day", "S_norm", "R_norm"]].copy()
    df_states.columns = ["q_gr4j_mm", "S_norm_gr4j", "R_norm_gr4j"]
    df_states.to_csv(OUT_STATES)
    log.info(f"  {OUT_STATES.name}  (estados para merge D6)")

    # ── 8. Guardar parámetros + métricas DSST ────────────────────────────────
    def _fmt_metrics(m: dict) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) and not np.isnan(v) else None)
                for k, v in m.items() if k != "N"} | {"N": m.get("N")}

    params_yaml = {
        "model"           : "GR4J",
        "reference"       : "Perrin et al. (2003) doi:10.1016/S0022-1694(03)00225-7",
        "calibration_date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "basin"           : "Chancay-Huaral",
        "area_km2"        : float(AREA_KM2),
        "q_station"       : Q_COL,
        "pr_source"       : "PISCOp v3.0 — B2_pisco_v3_basin_mean.csv (D050)",
        "pet_source"      : ("B2_pet_era5pm_1981_2025.csv — "
                             "hscal 1981-2020 + ERA5 pev bias-corrected 2021-2025 (script 08f)"),
        "objective"       : "0.4×NSE + 0.3×KGE + 0.2×NSE_log − 0.1×|PBIAS_Q90|/100 (D052)",
        "optimizer"       : "differential_evolution (seed=42, maxiter=300) + Nelder-Mead polish",
        "parameters"      : {
            "X1": round(float(X1), 3),
            "X2": round(float(X2), 5),
            "X3": round(float(X3), 3),
            "X4": round(float(X4), 4),
        },
        "dsst_gate"       : "PASS" if gate_pass else "FAIL",
        "dsst_thresholds" : {
            "NSE_DSST": NSE_THR_DSST,
            "NSE_SST" : NSE_THR_SST,
            "NSE_cal" : NSE_THR_CAL,
        },
        "experiments": {
            r["label"][:2]: {
                "description": r["label"],
                "params": {
                    "X1": round(float(r["params"][0]), 3),
                    "X2": round(float(r["params"][1]), 5),
                    "X3": round(float(r["params"][2]), 3),
                    "X4": round(float(r["params"][3]), 4),
                },
                "metrics_cal": _fmt_metrics(r["metrics_cal"]),
                "metrics_val": _fmt_metrics(r["metrics_val"]) if r["metrics_val"] else None,
            }
            for r in results
        },
    }
    OUT_PARAMS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PARAMS, "w", encoding="utf-8") as fh:
        yaml.dump(params_yaml, fh, allow_unicode=True, default_flow_style=False,
                  sort_keys=False)
    log.info(f"  {OUT_PARAMS.name}")

    # ── 9. Figura ────────────────────────────────────────────────────────────
    make_dsst_figure(results, Q_opt, opt_idx, dry_mask, wet_all)

    log.info("\n" + "=" * 70)
    log.info("=== GR4J DSST COMPLETO ===")
    log.info(f"  Gate DSST   : {'PASS ✓' if gate_pass else 'FAIL ✗'}")
    log.info(f"  E4 params   : X1={X1:.1f}  X2={X2:.4f}  X3={X3:.1f}  X4={X4:.3f}")
    log.info(f"  Reconstrucción: {OUT_Q_SIM.name}  (Q_mean={Q_hist_m3s.mean():.2f} m³/s)")
    log.info(f"  Parámetros  : {OUT_PARAMS.name}")
    log.info(f"  Figura      : {OUT_FIG.name}")
    log.info("=" * 70)
    log.info("Siguiente: tests/test_no_leakage.py — verificar ausencia de data leakage")


if __name__ == "__main__":
    main()
