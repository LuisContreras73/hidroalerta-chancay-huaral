#!/usr/bin/env python3
"""
Script para cancelar programáticamente todas las tareas activas de GEE
(en estado READY o RUNNING) en el proyecto actual.
"""
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("cancel_tasks")

def cancel_all_tasks():
    try:
        import ee
        ee.Initialize(project="ana-chancay-huaral",
                      opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception as e:
        log.error(f"Error al inicializar GEE: {e}")
        return

    log.info("Obteniendo lista de tareas en Earth Engine...")
    tasks = ee.data.listOperations()
    
    active_tasks = []
    for task in tasks:
        # Los estados activos en la API de operaciones son 'PENDING' o 'RUNNING'
        state = task.get("metadata", {}).get("state")
        if state in ("PENDING", "RUNNING"):
            active_tasks.append(task)
            
    n_active = len(active_tasks)
    if n_active == 0:
        log.info("No se encontraron tareas activas o en cola para cancelar.")
        return
        
    import concurrent.futures
    log.info(f"Se encontraron {n_active} tareas activas/en cola. Cancelando con ThreadPoolExecutor (50 workers)...")
    
    def cancel_one(task):
        name = task.get("name")
        description = task.get("metadata", {}).get("description", "Sin descripcion")
        try:
            ee.data.cancelOperation(name)
            return True
        except Exception as e:
            log.warning(f"No se pudo cancelar la tarea {description}: {e}")
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        results = list(executor.map(cancel_one, active_tasks))
        
    cancelled = sum(1 for r in results if r)
    log.info(f"✅ Proceso terminado. Se cancelaron {cancelled} tareas activas en GEE.")

if __name__ == "__main__":
    cancel_all_tasks()
