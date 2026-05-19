"""
conftest.py — Configuración compartida de pytest para el benchmark.
"""

import sys
from pathlib import Path

# Añadir la raíz del proyecto al path para que los imports funcionen
sys.path.insert(0, str(Path(__file__).parent.parent))
