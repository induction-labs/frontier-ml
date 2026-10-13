"""Best-effort operator progress. No output unless the trusted caller supplies a path."""
import json
from pathlib import Path
import tempfile
import threading
import time

_lock = threading.Lock()
_last = {}


def write_json(path, value):
    """Atomic replacement; observability must never turn a valid evaluation into a failure."""
    temporary = None
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".progress-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    except (OSError, ValueError, TypeError):
        pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def progress(config, stage, *, completed=None, total=None, message=None):
    path = config.get("progress_path")
    if not path:
        return
    with _lock:
        now = time.monotonic()
        previous_stage, previous_time = _last.get(str(path), (None, 0))
        if stage == previous_stage and now - previous_time < 1 and completed != total:
            return
        _last[str(path)] = (stage, now)
        value = {"stage": stage, "updated_at": time.time()}
        if completed is not None:
            value["completed"] = completed
        if total is not None:
            value["total"] = total
        if message:
            value["message"] = message
        write_json(path, value)
