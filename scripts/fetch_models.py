"""Download and verify the pinned Tesseract models."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

TESSDATA = Path(__file__).resolve().parents[1] / "src" / "docfgp" / "tessdata"
RAW_URL = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/{revision}/{name}"


def _digest(path: Path) -> str:
    """Hash a model file without reading it into memory at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, target: Path, sha256: str) -> None:
    """Replace the target only after the download matches its pinned digest."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        partial = Path(stream.name)

    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
        if _digest(partial) != sha256:
            raise SystemExit(f"{url} does not match its pinned SHA-256 digest")
        partial.chmod(0o644)
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    """Place every pinned model in the bundled model directory.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    manifest = json.loads((TESSDATA / "models.json").read_text(encoding="utf-8"))

    for name, model in manifest["models"].items():
        target = TESSDATA / f"{name}.traineddata"
        if target.is_file() and _digest(target) == model["sha256"]:
            continue

        url = RAW_URL.format(revision=manifest["revision"], name=f"{name}.traineddata")
        _fetch(url, target, model["sha256"])
        print(target)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
