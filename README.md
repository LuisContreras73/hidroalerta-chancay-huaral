# HidroAlerta Chancay-Huaral

Estación **IoT + IA** de pronóstico de caudal y **alerta temprana** para la cuenca Chancay-Huaral (Perú).
*I Concurso de Ciencia y Tecnología para la Seguridad Hídrica en la Cuenca Chancay-Huaral — ANA 2026.*

## Qué es

Un sistema que combina un **nodo IoT de campo** (ESP32-SUPERMINI + LoRa, sensores de nivel, temperatura y suelo, autónomo con panel solar y batería) con un **motor de IA** que pronostica el caudal diario a **1–14 días** con incertidumbre, emite **alerta temprana probabilística** de crecidas y se **verifica in-situ** (el nivel medido cae dentro de la banda pronosticada).

## Resultados (partición de prueba, años nunca vistos)

- **NSE a 14 días = 0,755** con el modelo local (Temporal Fusion Transformer canónico + núcleo GRU), frente a 0,494 del modelo previo.
- **Banda de incertidumbre ~4× más ajustada** que persistencia/climatología (CRPS a 14 d: 1,23 vs 4,57 vs 4,45).
- **Alerta dura** accionable a **1–3 días**; **triage de riesgo** a 5–14 días (AUC ≈ 0,87).
- Arquitectura **híbrida**: modelo fundacional zero-shot (Chronos-2) para el nowcast + modelo local para el subestacional, unidos por una compuerta.
- Validación: **split temporal** (train ≤2022 · val 2023 · test 2024–25), CRPS, KGE, block-bootstrap, N_eff, calibración (reliability/PIT).

### Limitaciones (declaradas)

Los picos extremos se sub-estiman a plazo largo (la alerta útil de umbral duro es a 1–3 días). Validación *single-basin* con pocos eventos efectivos (N_eff pequeño); el escalado multi-cuenca (CAMELS-PE) está en curso. No se mide calidad química (solo temperatura y turbidez/suelo). El nodo es un prototipo de una estación y la curva nivel→caudal es aproximada.

## Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| `src/` | Módulos: construcción de *features*, *target*, modelos (TFT, GRU, fundacional) y utilidades. |
| `scripts/` | Pipeline reproducible: `01_bronze` (ingesta) → `02_silver` → `03_gold` → `04_hydro` → `05_models` → `06_eval`; `08_gee` (Google Earth Engine). |
| `configs/` | Configuraciones de modelos y experimentos. |
| `data/processed/` | Datos procesados ligeros. Los crudos pesados están en Drive (ver `FUENTES.md`). |
| `hardware/` | Nodo IoT: `cad/` (modelos 3D `.step`/`.wrl`) y `kicad/` (esquemas eléctricos). |
| `docs/` | Documentación adicional. |

## Cómo ejecutar

1. **Entorno:** Python 3.13 + PyTorch. `pip install -r requirements.txt`.
2. **Datos:** descargar los crudos desde el enlace de Drive indicado en `FUENTES.md` y colocarlos en `data/`.
3. **Pipeline:** ejecutar los scripts en orden `01_bronze` → `06_eval`.

## Datos y licencias de fuentes

Ver **`FUENTES.md`**. Se usan datos abiertos: precipitación/temperatura **PISCO** (SENAMHI/IGP), reanálisis **ERA5-Land** (ECMWF/Copernicus), caudal **ANA/SNIRH**, índices **ENSO** (NOAA), y capas de **Google Earth Engine**.

## Equipo

- **Luis A. Contreras** — Ing. Ambiental · Deep Learning
- **Diego Mijahuanca** — Ing. Ambiental · análisis de datos
- **Samir Suarez** — Ing. Mecatrónica · nodo IoT
- Asesor: **Ing. Jorge Zafra** — Hidrólogo (UTEC)

Universidad de Ingeniería y Tecnología (**UTEC**), 2026.

## Licencia

Código bajo licencia **MIT** (ver `LICENSE`). Los datos de terceros conservan sus licencias originales (ver `FUENTES.md`).
