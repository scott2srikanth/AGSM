import json
import os
import tempfile
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False
    ) as temporary:
        try:
            json.dump(value, temporary, indent=2, sort_keys=True, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            os.replace(temporary.name, path)
        finally:
            Path(temporary.name).unlink(missing_ok=True)
