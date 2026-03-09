"""
Shared import configuration for approximate-multiplier scripts.
Adds HPC directory to Python path so scripts can import as if running from HPC.
"""
import sys
import os

# Add HPC directory to path
_hpc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'HPC'))
if _hpc_path not in sys.path:
    sys.path.insert(0, _hpc_path)