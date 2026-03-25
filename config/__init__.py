"""
Path resolution strategy:
1. Use ./data if it contains the expected JSONL files
2. Fall back to DATA_DIR from config/settings.local.py
3. Raise a clear error if neither works
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_DATA_DIR = PROJECT_ROOT / 'data'
_DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs'

_EXPECTED_FILE = 'train_label.jsonl'


def get_data_dir() -> str:
    # 1. Try local ./data folder
    if (_DEFAULT_DATA_DIR / _EXPECTED_FILE).exists():
        return str(_DEFAULT_DATA_DIR)

    # 2. Try settings.local.py
    try:
        from config.settings_local import DATA_DIR
        if DATA_DIR and Path(DATA_DIR, _EXPECTED_FILE).exists():
            return str(DATA_DIR)
    except ImportError:
        pass

    raise FileNotFoundError(
        f"Cannot find '{_EXPECTED_FILE}'.\n"
        f"  Option 1: Place JSONL files in '{_DEFAULT_DATA_DIR}'\n"
        f"  Option 2: Set DATA_DIR in 'config/settings.local.py'"
    )


def get_output_dir() -> str:
    try:
        from config.settings_local import OUTPUT_DIR
        if OUTPUT_DIR:
            Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            return str(OUTPUT_DIR)
    except ImportError:
        pass

    _DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return str(_DEFAULT_OUTPUT_DIR)
