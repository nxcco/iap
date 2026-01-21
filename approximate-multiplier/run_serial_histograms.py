import yaml
import os
import subprocess
import sys
import _import_config

# Config laden
config_path = os.path.join(os.path.dirname(__file__), '../config_primr8.yaml')
with open(config_path, 'r') as f:
    full_yaml = yaml.safe_load(f)

# Pfad zur Config extrahieren
if 'afpm_analysis' in full_yaml and 'configurations' in full_yaml['afpm_analysis']:
    configs = full_yaml['afpm_analysis']['configurations']
else:
    # Fallback, falls flache Struktur
    configs = full_yaml

# Alle relevanten Configs finden
targets = [k for k in configs.keys() if k.startswith('FPM_T6')]
targets.sort()

print(f"Found {len(targets)} configurations. Starting serial processing...")

plotter_script = os.path.join(os.path.dirname(__file__), 'bias_plotter.py')

for i, config_name in enumerate(targets):
    print(f"\n[{i+1}/{len(targets)}] Processing {config_name}...")
    
    cmd = [sys.executable, plotter_script, '--config', config_name, '--samples', '2000000']
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR processing {config_name}: {e}")
        continue

print("\nAll done!")