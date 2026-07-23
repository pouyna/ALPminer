"""Shared utilities: logging, retries, atomic file writes, hashing."""

from __future__ import annotations

import hashlib
import logging
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterable, Type, TypeVar

log = logging.getLogger("alpminer")

T = TypeVar("T")


def setup_logging(log_dir: Path, verbose: bool = False) -> None:
    """Log DEBUG to file, INFO (or DEBUG with --verbose) to console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    fh = logging.FileHandler(log_dir / "alpminer.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    log.addHandler(fh)

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    log.addHandler(ch)


class RetryError(RuntimeError):
    """Raised when all retry attempts are exhausted."""


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Seconds requested by a ``Retry-After`` header on the exception's HTTP
    response (the integer-seconds form; the HTTP-date form is ignored and we
    fall back to backoff). ``None`` when there is no such header -- lets a 429
    from OpenAlex/Unpaywall/an LLM API tell us exactly how long to wait."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def with_retries(
    fn: Callable[[], T],
    *,
    attempts: int = 4,
    base_delay: float = 2.0,
    max_delay: float = 90.0,
    retry_on: tuple[Type[BaseException], ...] = (Exception,),
    give_up_on: tuple[Type[BaseException], ...] = (),
    desc: str = "operation",
) -> T:
    """Call fn(), retrying with exponential backoff + jitter on retry_on.

    A ``Retry-After`` header on the failed response (e.g. an HTTP 429) is
    honored in preference to the computed backoff. Exceptions in give_up_on
    are re-raised immediately (no retry).
    """
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except give_up_on:
            raise
        except retry_on as exc:  # noqa: PERF203
            last_exc = exc
            if attempt == attempts:
                break
            retry_after = _retry_after_seconds(exc)
            if retry_after is not None:
                delay = min(max_delay, retry_after)   # server told us how long
            else:
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                delay += random.uniform(0, delay * 0.25)
            # INFO (not DEBUG) so a long rate-limit wait is visible in the GUI
            # console and the CLI, instead of looking like the job has hung.
            log.info("%s failed (attempt %d/%d): %s -- retrying in %.0fs",
                     desc, attempt, attempts, exc, delay)
            time.sleep(delay)
    raise RetryError(f"{desc} failed after {attempts} attempts: {last_exc}") from last_exc


def atomic_write_text(path: Path, text: str) -> None:
    """Write text to path atomically (tmp file in same dir + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def human_int(n: int) -> str:
    return f"{n:,}"


def chunked(seq: list, size: int) -> Iterable[list]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]
