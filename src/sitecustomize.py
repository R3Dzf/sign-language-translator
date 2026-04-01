"""
Auto-loaded by Python when this folder is on sys.path.
It suppresses noisy third-party warnings before the rest of the app imports.
"""

from src.runtime_setup import suppress_runtime_warnings


suppress_runtime_warnings()
