import csv
import sys
from pathlib import Path

import pandas as pd


def build_id(row: pd.Series):
    year = int(row["identifier1"].split("_")[1])
    preffix = "old" if year < 1900 else "new"
    row["identifier1"] = f"{preffix}_{row['identifier1']}"

    year = int(row["identifier2"].split("_")[1])
    preffix = "old" if year < 1900 else "new"
    row["identifier2"] = f"{preffix}_{row['identifier2']}"


def create_grouping(data: dict, prompts):
    for prompt in prompts:
        full_data = data[prompt]
        full_data.apply(lambda row: build_id(row), axis=1)
