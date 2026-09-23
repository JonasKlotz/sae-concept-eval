"""Default filesystem locations for scripts that are not driven by Hydra.

Both roots can be overridden with the ``OUTPUTS_DIR`` / ``DATA_ROOT`` env vars
(or a ``.env`` file in the repo root), mirroring ``config/base.yaml``.
"""

import os
from pathlib import Path

import rootutils

PROJECT_ROOT = Path(rootutils.setup_root(__file__, dotenv=True, pythonpath=False, cwd=False))
OUTPUTS_ROOT = Path(os.environ.get("OUTPUTS_DIR", PROJECT_ROOT / "outputs"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data"))
METRICS_ROOT = OUTPUTS_ROOT / "metrics"
FIGURES_ROOT = OUTPUTS_ROOT / "figures"
