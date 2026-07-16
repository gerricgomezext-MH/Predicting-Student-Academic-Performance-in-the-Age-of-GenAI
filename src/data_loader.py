"""Data loading utilities.

Loads the raw AI-impact student performance dataset used throughout the
capstone project.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai_impact_student_performance_dataset.csv"


def load_dataset(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset from ``path`` into a pandas DataFrame.

    Parameters
    ----------
    path:
        Path to the CSV file. Defaults to ``data/raw/ai_impact_student_performance_dataset.csv``.

    Returns
    -------
    pandas.DataFrame
        The raw, unmodified dataset.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Make sure the CSV file is available "
            "in the data/raw/ directory."
        )
    return pd.read_csv(path)
