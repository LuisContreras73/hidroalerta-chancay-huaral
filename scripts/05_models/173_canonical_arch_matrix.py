#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 173 — MATRIZ de ablaciones ARQUITECTÓNICAS sobre el TFT canónico (145).

Objetivo (ablación correcta, exigencia Q1): partir del canónico como BASE y probar
las mejoras que la literatura reciente demostró, aplicando **un solo cambio a la vez**
(Tier-1) y luego **combinando los ganadores** (Tier-2). Modelo COMPOSICIONAL: cada eje
se enciende independientemente, así el mismo código corre base, cambios simples y combos.

Ejes (cada uno con su nivel base = el canónico de Lim et al. 2021):
  rec      recurrencia enc/dec : lstm(base) | bilstm | gru | bigru
  act      activación en TODAS las GRN (→ afecta VSN, enrich, ff) : elu(base) | gelu | silu | mish
  gate     compuerta GLU        : glu(base) | swiglu | geglu
  grnnorm  norm DENTRO de la GRN : layer(base) | rms
  seqnorm  norm a nivel secuencia: layer(base) | sandwich (añade RMS pre-atención + post, ENCIMA)
  qk       norm en atención     : none(base) | qknorm (RMSNorm en Q,K por cabeza)

Nota de diseño (lo que pidió el usuario): las normas RMS del eje `seqnorm=sandwich` se
AÑADEN encima (pre+post), no reemplazan; el resto (activación, celda recurrente, gate)
son swaps inherentes. Cabeza cuantílica (p10/p50/p90) y pérdida (pinball) intactas:
esto aísla ARQUITECTURA. La familia de cambios de cabeza/pérdida vive en 143/154.

Comparación JUSTA: MISMO harness (T.train_es), datos, split (train ≤2022 · val 2023 ·
test 2024-25), hparams (estudio intv_h14), seeds 0/1/2. NO Optuna (ablación a HP fijos).
Baseline de referencia: TFT-canónico (145). Todo append-only, artefactos 173_*.

Uso (.venv313, SÓLO con GPU libre):
    python scripts/05_models/173_canonical_arch_matrix.py --smoke      # valida construcción
    python scripts/05_models/173_canonical_arch_matrix.py --tier 1     # 11 cambios + base
    python scripts/05_models/173_canonical_arch_matrix.py --tier 2     # combos (lee tier1)
Salida: outputs/ml_Q/173_matrix_tier1.csv · 173_matrix_tier2.csv (+ checkpoints 173_*.pt)
"""
import argparse
import importlib.util
import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "outputs/ml_Q"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("matrix173")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _load(mod, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / mod)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


C = _load("scripts/05_models/145_canonical_tft.py", "c145")     # backbone canónico
T = C.T                                                          # harness de 114 (train_es, seqs_for_enc)
R = C.R
DEV = C.DEV
Q90 = R.Q90
H = 14                       # horizonte de entrenamiento (se sobre-escribe con --H)
LEADS = [1, 3, 7, 14]        # se recalcula según H


def leads_for(h):
    base = [l for l in (1, 3, 7, 14, 21, 30) if l <= h]
    if h not in base:
        base.append(h)
    return base


ACT = {"elu": F.elu, "gelu": F.gelu, "silu": F.silu, "mish": F.mish, "relu": F.relu}


def mknorm(kind, d):
    return nn.RMSNorm(d) if kind == "rms" else nn.LayerNorm(d)


# ── componentes configurables ─────────────────────────────────────────────────
class GRNx(nn.Module):
    """GRN de Lim et al. con activación, compuerta y norma configurables.
    gate: glu = a·σ(b) (canónico) | swiglu = a·silu(b) | geglu = a·gelu(b)."""

    def __init__(self, d, drop=0.1, d_ctx=None, act="elu", gate="glu", norm="layer"):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.ctx = nn.Linear(d_ctx, d, bias=False) if d_ctx else None
        self.fc2 = nn.Linear(d, d)
        self.act = ACT[act]
        self.gate_kind = gate
        self.gate = nn.Linear(d, 2 * d)
        self.drop = nn.Dropout(drop)
        self.norm = mknorm(norm, d)

    def forward(self, a, c=None):
        eta = self.fc1(a)
        if self.ctx is not None and c is not None:
            eta = eta + self.ctx(c)
        eta = self.fc2(self.act(eta))
        x = self.gate(self.drop(eta))
        h = x.shape[-1] // 2
        b0, b1 = x[..., :h], x[..., h:]
        if self.gate_kind == "swiglu":
            glu = b0 * F.silu(b1)
        elif self.gate_kind == "geglu":
            glu = b0 * F.gelu(b1)
        else:
            glu = b0 * torch.sigmoid(b1)
        return self.norm(a + glu)


class VSNx(nn.Module):
    """Variable Selection Network con GRNx (hereda act/gate/norm → 'GRN mod = VSN mod')."""

    def __init__(self, n_vars, d, drop=0.1, act="elu", gate="glu", norm="layer"):
        super().__init__()
        self.n = n_vars
        self.proj = nn.ModuleList([nn.Linear(1, d) for _ in range(n_vars)])
        self.var_grn = nn.ModuleList([GRNx(d, drop, act=act, gate=gate, norm=norm) for _ in range(n_vars)])
        self.sel = nn.Linear(n_vars * d, n_vars)

    def forward(self, x):
        feats = [self.var_grn[i](self.proj[i](x[..., i:i + 1])) for i in range(self.n)]
        stacked = torch.stack(feats, dim=-2)
        flat = stacked.flatten(-2)
        w = torch.softmax(self.sel(flat), dim=-1).unsqueeze(-1)
        return (stacked * w).sum(-2), w.squeeze(-1)


class RecCore(nn.Module):
    """Núcleo recurrente enc/dec: LSTM/GRU, uni/bidireccional. Encapsula el traspaso de
    estado enc→dec y proyecta la salida a `hid` (bidir=2·hid→hid). El decoder procesa
    covariables FUTURAS conocidas (calendario), no el objetivo → bidir no filtra futuro."""

    def __init__(self, kind, hid):
        super().__init__()
        self.kind = kind
        self.bidir = kind.startswith("bi")
        self.is_lstm = ("gru" not in kind)
        cell = nn.LSTM if self.is_lstm else nn.GRU
        self.enc = cell(hid, hid, batch_first=True, bidirectional=self.bidir)
        self.dec = cell(hid, hid, batch_first=True, bidirectional=self.bidir)
        self.proj = nn.Linear(2 * hid, hid) if self.bidir else nn.Identity()

    def forward(self, sp, sf):
        if self.is_lstm:
            e, (hN, cN) = self.enc(sp)
            d, _ = self.dec(sf, (hN, cN))
        else:
            e, hN = self.enc(sp)
            d, _ = self.dec(sf, hN)
        return self.proj(e), self.proj(d)


class QKNormIMHA(C.IMHA):
    """IMHA canónica + RMSNorm en Q y K por cabeza (estabiliza el score; QK-norm)."""

    def __init__(self, d, heads, drop=0.1):
        super().__init__(d, heads, drop)
        self.rq = nn.RMSNorm(self.dh); self.rk = nn.RMSNorm(self.dh)

    def forward(self, q, k, v, mask=None):
        B, Lq, _ = q.shape; Lk = k.shape[1]
        qh = self.rq(self.q(q).view(B, Lq, self.h, self.dh)).transpose(1, 2)
        kh = self.rk(self.k(k).view(B, Lk, self.h, self.dh)).transpose(1, 2)
        vs = self.v(v).unsqueeze(1)
        sc = qh @ kh.transpose(-1, -2) / self.dh ** 0.5
        if mask is not None:
            sc = sc.masked_fill(mask, -1e9)
        att = torch.softmax(sc, dim=-1)
        return self.o((self.drop(att) @ vs).mean(1)), att.mean(1)


class TFTCanonMatrix(C.TFTCanonical):
    """TFT canónico con los 6 ejes conmutables. Con todos en base ≈ 145 (misma arquitectura)."""

    def __init__(self, n_past, n_future, hid=128, heads=2, H=14, drop=0.2,
                 q_mu=0.0, q_sd=1.0, rec="lstm", act="elu", gate="glu",
                 grnnorm="layer", seqnorm="layer", qk="none", **_):
        nn.Module.__init__(self)
        self.H = H
        self.register_buffer("q_mu", torch.tensor(float(q_mu)))
        self.register_buffer("q_sd", torch.tensor(float(q_sd)))
        gk = dict(act=act, gate=gate, norm=grnnorm)
        self.vsn_past = VSNx(n_past + 1, hid, drop, **gk)
        self.vsn_fut = VSNx(n_future, hid, drop, **gk)
        self.static = nn.Parameter(torch.zeros(1, hid))
        self.rec = RecCore(rec, hid)
        self.gate_lstm = nn.Linear(hid, 2 * hid)
        # seqnorm: layer(base) | rms = REEMPLAZA las LayerNorm de secuencia por RMSNorm |
        # sandwich = rms + AÑADE una norm pre-atención (única adición, con criterio).
        seq_rms = seqnorm in ("rms", "sandwich")
        self.norm_lstm = mknorm("rms" if seq_rms else "layer", hid)
        self.enrich = GRNx(hid, drop, d_ctx=hid, **gk)
        self.attn = QKNormIMHA(hid, heads, drop) if qk == "qknorm" else C.IMHA(hid, heads, drop)
        self.gate_att = nn.Linear(hid, 2 * hid)
        self.norm_att = mknorm("rms" if seq_rms else "layer", hid)
        self.pre_attn = mknorm("rms", hid) if seqnorm == "sandwich" else None
        self.ff = GRNx(hid, drop, **gk)
        self.head = nn.Linear(hid, 3)
        self.cfg = dict(rec=rec, act=act, gate=gate, grnnorm=grnnorm, seqnorm=seqnorm, qk=qk)

    def _gan(self, x, resid, gate, norm):
        g = gate(x); h = g.shape[-1] // 2
        return norm(resid + g[..., :h] * torch.sigmoid(g[..., h:]))

    def forward(self, x_past, q_past, x_future):
        B = x_past.shape[0]
        qn = ((q_past - self.q_mu) / self.q_sd).unsqueeze(-1)
        past = torch.cat([x_past, qn], dim=-1) if x_past.shape[-1] != 0 else qn
        sp, _ = self.vsn_past(past)
        sf, _ = self.vsn_fut(x_future)
        ctx = self.static.expand(B, -1)
        e, d = self.rec(sp, sf)
        seq = self._gan(torch.cat([e, d], 1), torch.cat([sp, sf], 1), self.gate_lstm, self.norm_lstm)
        seq = self.enrich(seq, ctx.unsqueeze(1).expand(-1, seq.shape[1], -1))
        L = seq.shape[1]
        mask = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), 1).view(1, L, L)
        qkv = self.pre_attn(seq) if self.pre_attn is not None else seq
        a, _ = self.attn(qkv, qkv, qkv, mask=mask)
        seq = self._gan(a, seq, self.gate_att, self.norm_att)
        seq = self.ff(seq)
        dec = seq[:, -self.H:, :]
        raw = self.head(dec)
        p50 = raw[..., 0]
        p10 = p50 - F.softplus(raw[..., 1]); p90 = p50 + F.softplus(raw[..., 2])
        return torch.stack([p10, p50, p90], -1) * self.q_sd + self.q_mu


# ── diseño de la matriz ─────────────────────────────────────────────────────────
BASE = dict(rec="lstm", act="elu", gate="glu", grnnorm="layer", seqnorm="layer", qk="none")
AXES = {
    "rec":     ["bilstm", "gru", "bigru"],   # gru = REEMPLAZO; bi* = adición (más complejidad)
    "act":     ["gelu", "silu", "mish"],     # REEMPLAZO de la activación en las GRN
    "gate":    ["swiglu", "geglu"],          # REEMPLAZO de la compuerta GLU
    "grnnorm": ["rms"],                       # REEMPLAZO de LayerNorm por RMSNorm dentro de la GRN
    "seqnorm": ["rms", "sandwich"],           # rms = REEMPLAZO; sandwich = adición pre-atención (con criterio)
    "qk":      ["qknorm"],                     # adición de RMSNorm en Q,K (con criterio, estabiliza el score)
}


def cfg_with(**over):
    c = dict(BASE); c.update(over); return c


def tier1_configs():
    out = [("base", dict(BASE))]
    for axis, levels in AXES.items():
        for lv in levels:
            out.append((f"{axis}={lv}", cfg_with(**{axis: lv})))
    return out


def tag_of(name):
    return name.replace("=", "-").replace("+", "_").replace(":", "")


def run_config(name, cfg, data, bp, qmu, qsd, seeds=3, prefix="173"):
    Xtr, Xva, Xte, te, obs = data
    P = []
    for s in range(seeds):
        torch.manual_seed(s); np.random.seed(s)
        dl = DataLoader(TensorDataset(*Xtr), batch_size=bp["bs"], shuffle=True)
        mo = TFTCanonMatrix(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"],
                            H=H, drop=bp["drop"], q_mu=qmu, q_sd=qsd, **cfg).to(DEV)
        mo = T.train_es(mo, dl, Xva, bp["lr"], bp["wd"])
        torch.save(mo.state_dict(), OUT / f"{prefix}_{tag_of(name)}_seed{s}.pt")
        with torch.no_grad():
            P.append(mo(Xte[0].to(DEV), Xte[1].to(DEV), Xte[2].to(DEV)).cpu().numpy())
    PR = np.mean(P, 0)
    n_params = sum(p.numel() for p in mo.parameters())
    rows = []
    for h in LEADS:
        j = [i + h - 1 for i in te]
        o = np.array([obs[k] for k in j])
        rows.append(dict(variante=name, h=h, params=n_params, **C.met(o, PR[:, h - 1, :]), **cfg))
    log.info(f"✔ {name}  (params={n_params/1e3:.0f}K)")
    return rows


def load_data(bp):
    df = R.build(H); dts = df.index; obs = df["obs"].values; q = df["q"].values
    EM = 90
    tr = [i for i in range(EM, len(df) - H) if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i - EM:i + H]).all()]
    va = [i for i in range(EM, len(df) - H) if pd.Timestamp("2023-01-01") <= dts[i] <= pd.Timestamp("2023-12-31") and np.isfinite(q[i - EM:i + H]).all()]
    te = [i for i in range(EM, len(df) - H) if dts[i] >= pd.Timestamp("2024-01-01")]
    qtr = q[[i for i in range(len(df)) if dts[i] <= pd.Timestamp("2022-12-31") and np.isfinite(q[i])]]
    qmu, qsd = float(np.mean(qtr)), float(np.std(qtr) + 1e-6)
    mu = {"p": df[R.PAST].iloc[tr].values.mean(0), "f": df[R.FUT].iloc[tr].values.mean(0)}
    sd = {"p": df[R.PAST].iloc[tr].values.std(0) + 1e-6, "f": df[R.FUT].iloc[tr].values.std(0) + 1e-6}
    Xtr = T.seqs_for_enc(df, tr, bp["enc"], H, (mu, sd))
    Xva = T.seqs_for_enc(df, va, bp["enc"], H, (mu, sd))
    Xte = T.seqs_for_enc(df, te, bp["enc"], H, (mu, sd))
    return (Xtr, Xva, Xte, te, obs), qmu, qsd


def _axis_mean(sub, axis, lv):
    return sub[sub.variante == f"{axis}={lv}"].NSE.mean()


def tier2_configs(df1):
    """2 factores: ANCLA = mejor cambio individual (NSE media h∈{3,7,14}); se combina con el
    MEJOR nivel de cada OTRO eje (aunque ese nivel solo no bata la base — busca interacciones),
    más el combo de todos los ejes que sí superan la base individualmente."""
    rank_leads = [l for l in LEADS if l != 1] or LEADS   # subestacional: excluye h=1
    sub = df1[df1.h.isin(rank_leads)]
    base_nse = sub[sub.variante == "base"].NSE.mean()
    singles = sub[sub.variante.str.contains("=", na=False) & ~sub.variante.str.startswith("combo")]
    means = singles.groupby("variante").NSE.mean()
    anchor = means.idxmax()
    a_axis, a_lv = anchor.split("=", 1)
    log.info(f"Ancla Tier-2: {anchor} (NSE h3-14={means[anchor]:.3f} vs base {base_nse:.3f})")
    out, winners = [], {a_axis: a_lv}
    for axis, levels in AXES.items():
        if axis == a_axis:
            continue
        cand = {lv: _axis_mean(sub, axis, lv) for lv in levels}
        cand = {k: v for k, v in cand.items() if np.isfinite(v)}
        if not cand:
            continue
        best_lv = max(cand, key=cand.get)
        out.append((f"combo:{a_axis}={a_lv}+{axis}={best_lv}",
                    cfg_with(**{a_axis: a_lv, axis: best_lv})))
        if cand[best_lv] > base_nse + 1e-4:
            winners[axis] = best_lv
    if len(winners) >= 3:                       # combo de TODOS los ejes que superan base
        out.append(("combo:all-best", cfg_with(**winners)))
    return out


def append_reference(nuevo):
    cols = ["variante", "h", "params", "N", "NSE", "NSE_sqrt", "KGE", "MAE", "CRPS", "CSI", "POD", "FAR"]
    c145 = pd.read_csv(OUT / "145_canonical_tft.csv")
    ref = c145[(c145.model == "TFT-canónico") & c145.h.isin(LEADS)].copy()
    ref["variante"] = "canónico-145(ref)"; ref["params"] = np.nan
    piezas = [nuevo[[c for c in cols if c in nuevo.columns]],
              ref[[c for c in cols if c in ref.columns]]]
    return pd.concat(piezas, ignore_index=True).sort_values(["h", "variante"])


def parse_only(spec):
    """'seqnorm=rms,act=silu' → [(nombre, cfg), ...] reutilizando el mismo esquema de ejes."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok or "=" not in tok:
            continue
        axis, lv = tok.split("=", 1)
        out.append((tok, cfg_with(**{axis: lv})))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=1, choices=[1, 2])
    ap.add_argument("--smoke", action="store_true", help="1 seed, valida construcción+forward de todas las configs")
    ap.add_argument("--only", help="corre variantes puntuales (p.ej. 'seqnorm=rms') y las anexa al CSV Tier-1")
    ap.add_argument("--H", type=int, default=14, help="horizonte de entrenamiento (14 def; 30 = hasta ~1 mes)")
    a = ap.parse_args()
    global H, LEADS
    H = a.H
    LEADS = leads_for(H)
    sfx = "" if H == 14 else f"_H{H}"          # artefactos H=14 conservan nombres; H!=14 lleva sufijo
    C1, C2 = f"173{sfx}", f"173t2{sfx}"        # prefijos de checkpoints
    T1CSV = OUT / f"173_matrix_tier1{sfx}.csv"
    T2CSV = OUT / f"173_matrix_tier2{sfx}.csv"
    bp = optuna.load_study(study_name="intv_h14",
                           storage=f"sqlite:///{OUT / '114_intensive_h14.db'}").best_params
    data, qmu, qsd = load_data(bp)
    log.info(f"H={H} leads={LEADS} · train {len(data[0][0])} · test {len(data[3])} · hid={bp['hid']} enc={bp['enc']}")

    if a.smoke:
        Xtr, Xva, Xte, te, obs = data
        for name, cfg in tier1_configs():
            mo = TFTCanonMatrix(len(R.PAST), len(R.FUT), hid=bp["hid"], heads=bp["heads"],
                                H=H, drop=bp["drop"], q_mu=qmu, q_sd=qsd, **cfg).to(DEV)
            xb, qb, fb, yb = (t[:8].to(DEV) for t in Xtr)
            out = mo(xb, qb, fb)
            loss = F.l1_loss(out[..., 1], yb)
            loss.backward()
            npar = sum(p.numel() for p in mo.parameters())
            log.info(f"  OK {name:16s} out={tuple(out.shape)} params={npar/1e3:.0f}K")
        log.info("SMOKE OK — todas las configs construyen, forward y backward")
        return

    if a.only:
        rows = []
        for name, cfg in parse_only(a.only):
            rows += run_config(name, cfg, data, bp, qmu, qsd, prefix=C1)
        nuevo = pd.DataFrame(rows)
        if T1CSV.exists():
            prev = pd.read_csv(T1CSV)
            prev = prev[~prev.variante.isin(nuevo.variante.unique())]     # reemplaza si ya existía
            for c in prev.columns:
                if c not in nuevo.columns:
                    nuevo[c] = np.nan
            nuevo = pd.concat([prev, nuevo[prev.columns]], ignore_index=True)
        nuevo.sort_values(["h", "variante"]).to_csv(T1CSV, index=False)
        log.info(f"Anexado a {T1CSV.name}: {a.only}")
        return

    if a.tier == 1:
        rows = []
        for name, cfg in tier1_configs():
            rows += run_config(name, cfg, data, bp, qmu, qsd, prefix=C1)
        nuevo = pd.DataFrame(rows)
        tab = append_reference(nuevo) if H == 14 else nuevo.sort_values(["h", "variante"])
        tab.to_csv(T1CSV, index=False)
        log.info("\n" + tab[tab.h.isin([1, LEADS[-1]])].to_string(index=False))
        log.info(f"Guardado: {T1CSV.name}")
    else:
        df1 = pd.read_csv(T1CSV)
        cfgs = tier2_configs(df1)
        rows = []
        for name, cfg in cfgs:
            rows += run_config(name, cfg, data, bp, qmu, qsd, prefix=C2)
        nuevo = pd.DataFrame(rows)
        base1 = df1[df1.variante.isin(["base", "rec=gru", "canónico-145(ref)"])].copy()
        if "params" not in base1.columns:
            base1["params"] = np.nan
        cols = ["variante", "h", "params", "N", "NSE", "NSE_sqrt", "KGE", "MAE", "CRPS", "CSI", "POD", "FAR"]
        tab = pd.concat([nuevo[[c for c in cols if c in nuevo.columns]],
                         base1[[c for c in cols if c in base1.columns]]],
                        ignore_index=True).sort_values(["h", "variante"])
        tab.to_csv(T2CSV, index=False)
        log.info("\n" + tab[tab.h.isin([1, LEADS[-1]])].to_string(index=False))
        log.info(f"Guardado: {T2CSV.name}")


if __name__ == "__main__":
    main()
