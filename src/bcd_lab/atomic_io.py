"""Small atomic-file primitives shared by scientific artifact writers."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator, TextIO


def _temporary_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(name)


@contextmanager
def atomic_text_writer(path: str | Path) -> Iterator[TextIO]:
    """Yield a text stream that atomically replaces ``path`` on success."""
    target = Path(path)
    temporary = _temporary_path(target)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def atomic_binary_writer(path: str | Path) -> Iterator[BinaryIO]:
    """Yield a binary stream that atomically replaces ``path`` on success."""
    target = Path(path)
    temporary = _temporary_path(target)
    try:
        with temporary.open("wb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
