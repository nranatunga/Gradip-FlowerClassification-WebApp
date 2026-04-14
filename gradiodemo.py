"""Launch the Gradio app without an editable install (adds `src` to import path)."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from flower_classifier.app import main

if __name__ == "__main__":
    main()
