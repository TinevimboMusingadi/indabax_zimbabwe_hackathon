"""Simple context-manager timer for pipeline stages."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


@contextmanager
def timer(label: str) -> Generator[None, None, None]:
    """Log wall-clock time for a block of code."""
    start = time.perf_counter()
    logger.info("[START] %s", label)
    yield
    elapsed = time.perf_counter() - start
    logger.info("[DONE]  %s — %.1fs", label, elapsed)
