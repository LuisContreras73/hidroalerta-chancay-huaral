#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
VENV_PYTHON = ROOT / ".venv313" / "Scripts" / "python.exe"

def run_notebook_as_script(nb_path):
    nb_path = Path(nb_path)
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    code_blocks = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = cell['source']
            if isinstance(src, list):
                src = "".join(src)
            code_blocks.append(src)
    
    # Write to a temporary file in the notebooks directory so relative paths (like ../) work correctly
    temp_script = nb_path.parent / f"{nb_path.stem}_temp.py"
    
    # Setup non-interactive matplotlib backend and a mock display function
    header = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "display = lambda *args, **kwargs: print(*args) if args else None\n\n"
    )
    
    full_code = header + "\n\n# --- CELL ---\n\n".join(code_blocks)
    
    # Save the file
    temp_script.write_text(full_code, encoding='utf-8')
    print(f"Executing {nb_path.name} as script...")
    
    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    # Run the script using the virtual environment python
    res = subprocess.run([str(VENV_PYTHON), temp_script.name], cwd=str(nb_path.parent), capture_output=True, env=env)
    
    # Remove the temporary script
    if temp_script.exists():
        temp_script.unlink()
        
    stdout_decoded = res.stdout.decode('utf-8', errors='replace') if res.stdout else ""
    stderr_decoded = res.stderr.decode('utf-8', errors='replace') if res.stderr else ""
        
    if res.returncode != 0:
        print(f"Error executing {nb_path.name}:")
        print(stderr_decoded)
        return False
    else:
        print(f"Successfully executed {nb_path.name}.")
        if stdout_decoded.strip():
            print("Output:")
            print(stdout_decoded)
        return True

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    
    print("=== Re-generating EDA and Hydro Notebook Figures ===")
    notebooks = [
        NB_DIR / "03_pisco_preliminary_analysis.ipynb",
        NB_DIR / "04_observed_qaqc.ipynb",
        NB_DIR / "05_hydro_statistics.ipynb",
        NB_DIR / "06_gap_analysis.ipynb"
    ]
    
    for nb in notebooks:
        if not nb.exists():
            print(f"Notebook {nb.name} does not exist. Skipping.")
            continue
        success = run_notebook_as_script(nb)
        if not success:
            print(f"Failed to execute {nb.name}.")
            sys.exit(1)
            
    print("=== All figures successfully generated in outputs/figures/eda ===")

if __name__ == "__main__":
    main()
