import mimetypes
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pandas as pd

SUPPORTED_FORMATS = {"csv", "excel", "json", "parquet"}


class UnsupportedFormatError(ValueError):
    pass


class DatasetTooLargeError(ValueError):
    pass


def detect_format(path: str | Path, content_type: str | None = None) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    if suffix in ("csv", "txt", "tsv"):
        return "csv"
    if suffix in ("xlsx", "xls"):
        return "excel"
    if suffix == "json":
        return "json"
    if suffix == "parquet":
        return "parquet"
    if content_type:
        guessed = mimetypes.guess_extension(content_type) or ""
        return detect_format(guessed)
    raise UnsupportedFormatError(f"Could not determine dataset format for '{path}'")


def load_dataset(path: str | Path, fmt: str, max_rows: int | None = None) -> pd.DataFrame:
    if fmt not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(f"Unsupported format: {fmt}")

    if fmt == "csv":
        df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")
    elif fmt == "excel":
        df = pd.read_excel(path, sheet_name=0)
    elif fmt == "json":
        df = pd.read_json(path)
    elif fmt == "parquet":
        df = pd.read_parquet(path)
    else:  # pragma: no cover - guarded above
        raise UnsupportedFormatError(fmt)

    df.columns = [str(c).strip() for c in df.columns]

    if max_rows is not None and len(df) > max_rows:
        raise DatasetTooLargeError(f"Dataset has {len(df)} rows, exceeding the limit of {max_rows}")

    return df


def _safe_filename_from_url(url: str) -> str:
    name = Path(urlparse(url).path).name or "dataset"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name


def download_from_url(url: str, dest_dir: str | Path) -> tuple[Path, str]:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with httpx.stream("GET", url, follow_redirects=True, timeout=30.0) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        filename = _safe_filename_from_url(url)
        local_path = dest_dir / filename
        with open(local_path, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)

    fmt = detect_format(local_path, content_type=content_type)
    return local_path, fmt
