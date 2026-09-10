#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 146 — Tablas de parámetros por modelo (estilo PyTorch-Lightning ModelSummary).

Genera, para CADA arquitectura del zoo, la tabla submódulo × tipo × params × modo
(como el resumen que imprime Lightning al entrenar un TFT), SIN entrenar nada — solo
instancia con las dimensiones reales del proyecto y cuenta parámetros.

Cubre: TFT canónico (145), DLinear (145), RA-TFT gen4 (105), y las variantes de
normalización/atención (arch_variants: scalenorm, rezero, gatedrms, maskedself),
más las ya probadas del banco (sandwich, preRMS) via 143.

Salida: generacion_paper/tables/T3_model_summaries.md (todas las tablas) — para el
paper/PPT (comparar tamaño y composición de cada arquitectura de un vistazo).

Run en .venv313:
    python scripts/05_models/146_model_summaries.py
"""
import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "generacion_paper/tables/T3_model_summaries.md"


def _load(mod, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / mod)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


C = _load("scripts/05_models/145_canonical_tft.py", "c145")   # TFTCanonical, DLinear
V = _load("scripts/05_models/arch_variants.py", "av")         # variantes norm/attn
R = V.R                                                        # 105_ratft (RA-TFT base)

# dims reales del proyecto (estudio intv_h14: enc=90, hid=128, heads=2)
N_PAST, N_FUT, HID, HEADS, H, DROP, ENC = len(R.PAST), len(R.FUT), 128, 2, 14, 0.20, 90
QMU, QSD = 17.1, 15.0


def fmt(p: int) -> str:
    if p >= 1_000_000:
        return f"{p/1e6:.1f} M"
    if p >= 1000:
        return f"{p/1e3:.1f} K"
    return str(p)


def summary(model, titulo: str) -> str:
    filas = []
    for i, (n, m) in enumerate(model.named_children()):
        p = sum(x.numel() for x in m.parameters())
        filas.append((str(i), n, type(m).__name__, fmt(p)))
    # params sueltos a nivel módulo (nn.Parameter directos: RevIN.g/b, static, alpha…)
    directos = sum(x.numel() for n, x in model.named_parameters()
                   if "." not in n)
    total = sum(x.numel() for x in model.parameters())
    train = sum(x.numel() for x in model.parameters() if x.requires_grad)
    wn = max([len(r[1]) for r in filas] + [4])
    wt = max([len(r[2]) for r in filas] + [4])
    L = [f"### {titulo}", "",
         "```",
         f"{'':<3}| {'Name':<{wn}} | {'Type':<{wt}} | Params | Mode",
         "-" * (12 + wn + wt + 16)]
    for idx, n, t, p in filas:
        L.append(f"{idx:<3}| {n:<{wn}} | {t:<{wt}} | {p:>6} | train")
    if directos:
        L.append(f"{'—':<3}| {'(param. directos)':<{wn}} | {'Parameter':<{wt}} | {fmt(directos):>6} | train")
    L += ["-" * (12 + wn + wt + 16),
          f"{fmt(train):>10}  Trainable params",
          f"{fmt(total - train):>10}  Non-trainable params",
          f"{fmt(total):>10}  Total params",
          f"{total*4/1e6:>10.3f}  Total estimated params size (MB)",
          "```", ""]
    return "\n".join(L)


def build(cls, **extra):
    return cls(N_PAST, N_FUT, hid=HID, heads=HEADS, H=H, drop=DROP,
               enc=ENC, q_mu=QMU, q_sd=QSD, **extra)


def main():
    modelos = [
        ("TFT canónico (Lim 2021: VSN + GRN + Masked Interpretable MHA)",
         build(C.TFTCanonical)),
        ("DLinear (Zeng 2023)", build(C.DLinear)),
        ("RA-TFT (gen4, nuestro: cross-attn + gate GRN + LayerNorm + RevIN)",
         R.RATFT(N_PAST, N_FUT, hid=HID, heads=HEADS, H=H, drop=DROP)),
    ]
    for nombre, (Cls, kw) in V.VARIANTES.items():
        modelos.append((nombre, Cls(N_PAST, N_FUT, hid=HID, heads=HEADS, H=H,
                                     drop=DROP, **kw)))

    partes = ["# Tabla T3 — Resúmenes de parámetros por arquitectura",
              "",
              "> Estilo PyTorch-Lightning ModelSummary. Generado por `scripts/05_models/"
              "146_model_summaries.py` (solo instancia, no entrena). Dims reales: "
              f"n_past={N_PAST}, n_future={N_FUT}, hid={HID}, heads={HEADS}, H={H}, enc={ENC}.",
              ""]
    print(f"{'Modelo':<62} {'Total params':>12}")
    for nombre, mo in modelos:
        partes.append(summary(mo, nombre))
        tp = sum(x.numel() for x in mo.parameters())
        print(f"{nombre[:60]:<62} {fmt(tp):>12}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(partes), encoding="utf-8")
    print(f"\nGuardado: {OUT}")


if __name__ == "__main__":
    main()
