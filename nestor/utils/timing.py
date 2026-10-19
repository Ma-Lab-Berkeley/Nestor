"""Process-wide timing anchors.

``PROCESS_START`` is captured the first time this module is imported. To make
it reflect true process start (so MFU "including init" accounts for torch/cuda
import, dataset load, checkpoint resume and torch.compile), import this module
as early as possible in the entrypoint (see ``nestor/cli.py``).
"""

from __future__ import annotations

import time

# Monotonic clock anchor for the lifetime of this process (rank).
PROCESS_START: float = time.monotonic()
