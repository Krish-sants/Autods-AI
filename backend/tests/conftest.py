import os
import shutil
from pathlib import Path

# Must run before any `app.*` module is imported (conftest.py is always imported
# first by pytest), so the cached Settings singleton and the SQLAlchemy engine
# point at an isolated test database/data directory instead of the real one.
_TEST_DATA_DIR = Path(__file__).resolve().parent / ".tmp_test_data"
if _TEST_DATA_DIR.exists():
    shutil.rmtree(_TEST_DATA_DIR)
_TEST_DATA_DIR.mkdir(parents=True)

os.environ["DATA_DIR"] = str(_TEST_DATA_DIR)
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{(_TEST_DATA_DIR / 'test.db').as_posix()}"
os.environ.setdefault("GOOGLE_API_KEY", "")
os.environ.setdefault("TUNING_N_ITER", "3")  # keep CV/tuning fast in tests
os.environ.setdefault("CV_FOLDS", "2")

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_classification_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    tenure = rng.integers(1, 72, n)
    monthly_charges = rng.normal(70, 20, n).round(2)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n)
    signup_date = pd.date_range("2022-01-01", periods=n, freq="D")

    # Make churn meaningfully dependent on features so models can learn something real.
    score = (
        -0.05 * tenure
        + 0.02 * monthly_charges
        + np.where(contract == "month-to-month", 1.5, 0.0)
        + rng.normal(0, 1, n)
    )
    churn = (score > np.median(score)).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(n)],
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "contract": contract,
            "signup_date": signup_date.astype(str),
            "churn": churn,
        }
    )
    df.loc[df.sample(frac=0.05, random_state=1).index, "monthly_charges"] = np.nan
    return df


@pytest.fixture
def synthetic_regression_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 150
    size_sqft = rng.normal(1500, 400, n).round(0)
    bedrooms = rng.integers(1, 5, n)
    location = rng.choice(["downtown", "suburb", "rural"], n)
    price = size_sqft * 150 + bedrooms * 5000 + rng.normal(0, 10000, n)

    return pd.DataFrame(
        {"size_sqft": size_sqft, "bedrooms": bedrooms, "location": location, "price": price.round(2)}
    )


@pytest.fixture
def messy_csv_text() -> str:
    return (
        "Name, Age, Income\n"
        "Alice,30,50000\n"
        "Bob,-5,60000\n"
        "\n"
        "Carol,,\n"
        "Dave,40,70000\n"
        "Dave,40,70000\n"
    )
