"""Convert one Office file to PDF in a child process."""

import sys
from pathlib import Path

from ._native import office_to_pdf

if __name__ == "__main__":
    try:
        Path(sys.argv[2]).write_bytes(office_to_pdf(sys.argv[1]))
    except (OSError, RuntimeError) as exc:
        sys.exit(str(exc))
