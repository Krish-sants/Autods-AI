import json

import pandas as pd
import pytest

from app.ml.loaders import (
    UnsupportedFormatError,
    detect_format,
    download_from_url,
    load_dataset,
)


def test_detect_format_by_extension():
    assert detect_format("data.csv") == "csv"
    assert detect_format("data.xlsx") == "excel"
    assert detect_format("data.json") == "json"
    assert detect_format("data.parquet") == "parquet"


def test_detect_format_unsupported_raises():
    with pytest.raises(UnsupportedFormatError):
        detect_format("data.exe")


def test_load_csv_round_trip(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    loaded = load_dataset(path, "csv")
    assert list(loaded.columns) == ["a", "b"]
    assert len(loaded) == 3


def test_load_excel_round_trip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = tmp_path / "data.xlsx"
    df.to_excel(path, index=False)

    loaded = load_dataset(path, "excel")
    assert list(loaded.columns) == ["a", "b"]
    assert len(loaded) == 2


def test_load_json_round_trip(tmp_path):
    records = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    path = tmp_path / "data.json"
    path.write_text(json.dumps(records))

    loaded = load_dataset(path, "json")
    assert len(loaded) == 2


def test_load_parquet_round_trip(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    path = tmp_path / "data.parquet"
    df.to_parquet(path)

    loaded = load_dataset(path, "parquet")
    assert len(loaded) == 3


def test_load_dataset_max_rows_enforced(tmp_path):
    from app.ml.loaders import DatasetTooLargeError

    df = pd.DataFrame({"a": range(100)})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    with pytest.raises(DatasetTooLargeError):
        load_dataset(path, "csv", max_rows=10)


def test_download_from_url(tmp_path, monkeypatch):
    csv_bytes = b"a,b\n1,2\n3,4\n"

    class FakeResponse:
        headers = {"content-type": "text/csv"}

        def raise_for_status(self):
            pass

        def iter_bytes(self):
            yield csv_bytes

    class FakeStreamCtx:
        def __enter__(self):
            return FakeResponse()

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("httpx.stream", lambda *a, **k: FakeStreamCtx())

    local_path, fmt = download_from_url("https://example.com/dataset.csv", tmp_path)
    assert fmt == "csv"
    assert local_path.exists()
    assert local_path.read_bytes() == csv_bytes
