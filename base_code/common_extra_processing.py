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

    return row


def create_grouping(data: dict, prompts: list, logger):
    for prompt in prompts:
        if logger is not None:
            logger.info(f"creating grouping for model: {prompt}")

        full_data = data[prompt]
        data[prompt] = full_data.apply(lambda row: build_id(row), axis=1)

        if logger is not None:
            logger.info(f"grouping created for model: {prompt}")
