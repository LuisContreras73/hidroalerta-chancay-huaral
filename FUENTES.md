# Fuentes de datos y licencias

Todos los datos usados son de acceso abierto. Se citan aquí con su origen y licencia.

| Variable | Fuente | Licencia / condiciones |
|---|---|---|
| Precipitación diaria (PISCOp v3) | SENAMHI / IGP (Perú) | Uso abierto con atribución |
| Temperatura (PISCOt) | SENAMHI / IGP (Perú) | Uso abierto con atribución |
| Humedad de suelo, evapotranspiración (reanálisis) | ERA5-Land — ECMWF / Copernicus | Licencia Copernicus (atribución) |
| Caudal observado | ANA / SNIRH (Perú) | Datos públicos |
| Índices ENSO (ONI) | NOAA / CPC | Dominio público |
| Índice Niño costero (ICEN) | IGP / ENFEN (Perú) | Uso abierto con atribución |
| Capas satelitales y GIS (Sentinel-2, MODIS, DEM) | Google Earth Engine | Según cada colección (mayormente abiertas) |

## Datos en este repositorio

- `data/processed/` — versiones **procesadas y ligeras** suficientes para reproducir la evaluación
  (caudal observado, serie mensual, pronósticos multi-modelo, métricas por horizonte, benchmark).

## Datos crudos (pesados)

Los insumos crudos completos (series PISCO/ERA5 de 1981–2025, rásters GEE, checkpoints de modelos)
se distribuyen por Google Drive para no exceder los límites del repositorio:

> **Enlace Drive:** [COMPLETAR]

Descargar y colocar bajo `data/` siguiendo la estructura descrita en el README.
