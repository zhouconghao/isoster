"""QA plotting. Use ``matplotlib.use('Agg')`` at import time so the
orchestrator can run headlessly and in worker processes."""

import matplotlib

matplotlib.use("Agg")
