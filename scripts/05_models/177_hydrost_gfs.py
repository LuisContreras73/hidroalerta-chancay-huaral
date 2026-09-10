#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 177 — HydroST + GFS FUTURA por sub-cuenca (el "tenerlo ambos" espacial).

HydroST (78/139) es el modelo espacial multi-entidad; ya ingiere lluvia OBSERVADA por
sub-cuenca (pr_mm) en el encoder, pero NO tiene canal futuro. Aquí añadimos la lluvia
PRONOSTICADA GFS por sub-cuenca (B12, script 55) para el lead objetivo como entrada extra
por entidad al static-encoder -> CERO cambio de arquitectura (n_static += 2: pr_gfs_z + avail),
con channel-dropout (robustez: si GFS falta, cae al modo sin-GFS). Honesto (forecast, no obs)
y alineado (GFS emitido en t, lead L -> día t+L-1, coherente con 175). Mismo split/harness que 139.

Hipótesis: HydroST, diseñado para lo espacial, explota el GFS distribuido mejor que el TFT lumped.
Eval: HydroSTGFS-sano vs -caído (aísla el aporte within-run), leads 3/7/14, NSE + alerta (CSI/POD/FAR).

Salida: outputs/ml_Q/177_hydrost_gfs.csv
Run (.venv313, GPU): python scripts/05_models/177_hydrost_gfs.py   [--smoke]
"""
import importlib.util, logging, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, optuna
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent; OUT = ROOT / "outputs/ml_Q"
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("hgfs177"); optuna.logging.set_verbosity(optuna.logging.WARNING)
SMOKE = "--smoke" in sys.argv
spec = importlib.util.spec_from_file_location("h78", ROOT / "scripts/05_models/78_hydrost_Q.py")
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
DEV = H.DEVICE; QC = H.Q_CONV; Q90 = 40.89


def kge(o, p):
    r = np.corrcoef(o, p)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (p.std() / o.std() - 1) ** 2 + (p.mean() / o.mean() - 1) ** 2)


class GFSHydroDataset(H.HydroDataset):
    """Igual que HydroDataset pero aumenta x_static por entidad con [pr_gfs_z, avail] del lead L.
    train=True aplica channel-dropout (robustez); gfs_on=False fuerza modo caído (GFS a cero)."""
    def __init__(self, dyn, static, target, dates, target_dates, gfsz, gav, L,
                 train=False, cdrop=0.25, gfs_on=True):
        super().__init__(dyn, static, target, dates, target_dates)
        self.gfsz, self.gav, self.L = gfsz, gav, L
        self.train, self.cdrop, self.gfs_on = train, cdrop, gfs_on

    def __getitem__(self, i):
        idx, tgt = self.indices[i]
        x_dyn = torch.tensor(self.dyn[idx - H.ENC:idx].transpose(1, 0, 2), dtype=torch.float32)
        gfs = self.gfsz[idx, :, self.L - 1].astype(np.float32).copy()
        av = self.gav[idx, :, self.L - 1].astype(np.float32).copy()
        if not self.gfs_on:
            gfs[:] = 0.0; av[:] = 0.0
        elif self.train and np.random.rand() < self.cdrop:      # channel-dropout
            gfs[:] = 0.0; av[:] = 0.0
        static_aug = np.concatenate([self.static, gfs[:, None], av[:, None]], axis=1).astype(np.float32)
        return x_dyn, torch.tensor(static_aug), torch.tensor(float(tgt), dtype=torch.float32)


def main():
    bp = optuna.load_study(study_name="hydrost_arch", storage=f"sqlite:///{OUT/'115_hydrost.db'}").best_params
    H.ENC = bp["enc"]; H.LR1 = bp["lr1"]; H.LR2 = bp["lr2"]; H.WD = bp["wd"]; H.PAT = 25
    H.ALPHA = bp["alpha"]; H.ALERT_W = bp["alert_w"]
    df = H.load_data(); dyn, stat, tgt0, dates, scl = H.build_arrays(df); dpd = pd.DatetimeIndex(dates)
    date_pos = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    qmm = np.concatenate([[np.nan], tgt0[:-1]]).astype(np.float32)   # q_mm[idx]=q_next_1d[idx-1]
    ob = pd.read_csv(ROOT / "data/silver/snirh/S1_snirh_daily_q.csv", parse_dates=["date"])
    real = {d: v for d, v in zip(ob["date"], ob["q_santo_domingo_47e214d2"]) if np.isfinite(v)}

    # ── B12 -> GFS[n_days, n_ent, 14] alineado a HydroST (mismas fechas y orden de entidades) ──
    HH = 14; NE = len(H.ENTITIES); eidx = {e: i for i, e in enumerate(H.ENTITIES)}
    b = pd.read_csv(ROOT / "data/bronze/B12_gfs_daily_leads_subcuencas.csv", parse_dates=["init_date"])
    GFS = np.zeros((len(dates), NE, HH), np.float32); GAV = np.zeros((len(dates), NE, HH), np.float32)
    for (init, ent), g in b.groupby(["init_date", "entity_id"]):
        k = date_pos.get(pd.Timestamp(init)); j = eidx.get(ent)
        if k is None or j is None:
            continue
        for _, r in g.iterrows():
            L = int(r["lead"])
            if 1 <= L <= HH:
                GFS[k, j, L - 1] = r["pr_gfs_mm"]; GAV[k, j, L - 1] = 1.0
    trm = np.array([dpd[k] <= H.FT_END and GAV[k].any() for k in range(len(dates))])   # train stats
    gmu = np.zeros(NE); gsd = np.ones(NE)
    for j in range(NE):
        v = GFS[trm, j, :][GAV[trm, j, :] > 0]
        if v.size > 50:
            gmu[j] = v.mean(); gsd[j] = v.std() + 1e-6
    GFSz = ((GFS - gmu[None, :, None]) / gsd[None, :, None]) * GAV
    log.info(f"GFS alineado: {int((GAV[:,0,0]>0).sum())} inits con dato · {NE} entidades · "
             f"gmu costa(sub_634)={gmu[0]:.2f} vs cabecera(sub_656)={gmu[-1]:.2f} mm")

    pre = dpd[dpd <= H.PRETRAIN_END]; ftd = dpd[(dpd >= H.FT_START) & (dpd <= H.FT_END)]
    vad = dpd[(dpd >= H.VAL_START) & (dpd <= H.VAL_END)]; ted = dpd[dpd >= H.TEST_START]
    NST = H.N_STATIC + 2
    LEADS = [3] if SMOKE else [3, 7, 14]; NSEED = 1 if SMOKE else 3; E1, E2 = (2, 2) if SMOKE else (50, 150)
    rows = []
    for L in LEADS:
        tgtL = np.concatenate([qmm[L - 1:], [np.nan] * (L - 1)]).astype(np.float32)
        prs = []
        for s in range(NSEED):
            torch.manual_seed(s); np.random.seed(s)
            dsp = GFSHydroDataset(dyn, stat, tgtL, dates, pre, GFSz, GAV, L, train=True)
            dsf = GFSHydroDataset(dyn, stat, tgtL, dates, ftd, GFSz, GAV, L, train=True)
            dsv = GFSHydroDataset(dyn, stat, tgtL, dates, vad, GFSz, GAV, L, train=False, gfs_on=True)
            lp = DataLoader(dsp, batch_size=H.BATCH1, shuffle=True)
            lf = DataLoader(dsf, batch_size=bp["batch2"], shuffle=True)
            lv = DataLoader(dsv, batch_size=bp["batch2"], shuffle=False)
            m = H.HydroST(n_static=NST, hid=bp["hid"], heads=bp["heads"], layers=bp["layers"], dropout=bp["drop"]).to(DEV)
            m = H.train_phase1(m, lp, E1); m = H.train_phase2(m, lf, lv, E2); prs.append(m)
        for tag, gon in [("HydroSTGFS-sano", True), ("HydroSTGFS-caido", False)]:
            dst = GFSHydroDataset(dyn, stat, tgtL, dates, ted, GFSz, GAV, L, train=False, gfs_on=gon)
            ldt = DataLoader(dst, batch_size=256, shuffle=False)
            pdt = [pd.Timestamp(dates[idx + L - 1]) for idx, _ in dst.indices]
            yreal = np.array([real.get(d, np.nan) for d in pdt])
            allp = []
            for m in prs:
                m.eval(); pp = []
                with torch.no_grad():
                    for xd, xs, y in ldt:
                        pp.append(m(xd.to(DEV), xs.to(DEV)).cpu().numpy())
                allp.append(np.concatenate(pp, 0))
            PH = np.mean(allp, 0) / QC; mk = np.isfinite(yreal); o = yreal[mk]; p = PH[mk, 1]
            nse = 1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)
            oa, pa = o >= Q90, p >= Q90
            tp, fp, fn = int(np.sum(oa & pa)), int(np.sum(~oa & pa)), int(np.sum(oa & ~pa))
            csi = tp / (tp + fp + fn) if (tp + fp + fn) else 0
            pod = tp / (tp + fn) if (tp + fn) else 0; far = fp / (tp + fp) if (tp + fp) else 0
            rows.append(dict(modo=tag, lead=L, N=int(mk.sum()), NSE=round(float(nse), 3),
                             POD=round(pod, 3), CSI=round(csi, 3), FAR=round(far, 3)))
            log.info(f"{tag} lead {L}: NSE={nse:.3f} POD={pod:.3f} FAR={far:.3f} N={mk.sum()}")
        pd.DataFrame(rows).to_csv(OUT / "177_hydrost_gfs.csv", index=False)
    r = pd.DataFrame(rows); piv = r.pivot_table(index="lead", columns="modo", values="NSE")
    log.info("\nΔNSE (sano-caido, within-run):\n" + "\n".join(
        f"  lead {L}: {piv.loc[L,'HydroSTGFS-sano']-piv.loc[L,'HydroSTGFS-caido']:+.3f}" for L in LEADS))
    log.info("HYDROST_GFS_DONE; guardado 177_hydrost_gfs.csv")


if __name__ == "__main__":
    main()
