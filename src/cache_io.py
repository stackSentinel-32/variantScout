from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile


@contextmanager
def atomic_output(destination: Path):
    """Publish a finished file in one rename, preserving a prior entry on failure."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, suffix=destination.suffix, delete=False) as temp:
        pending = Path(temp.name)
    try:
        yield pending
        pending.replace(destination)
    finally:
        pending.unlink(missing_ok=True)
