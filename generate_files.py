from pathlib import Path
import logging

import pandas as pd


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

result_per_word = pd.DataFrame(
    {
        "word": [],
        "apd": [],
        # "jsd": [],
        # "clusters_to_freq1": [],
        # "clusters_to_freq2": [],
        "parameters": [],
    }
)

correlation_results = pd.DataFrame({"correlation": [], "parameters": []})

methods = ["wsbm"]
llms = ["llama3.1-8B", "mixtral-8xtb-v0.1"]
datasets = ["dwug_es", "dwug_en"]

path_to_folder = Path(f"./cv-apd-experiments-lscd")
if path_to_folder.exists() is False:
    path_to_folder.mkdir()

# for method in methods:
#     path_to_method = path_to_folder / Path(f"./{method}")

#     if path_to_method.exists() is False:
#         path_to_method.mkdir()

for llm in llms:
    path_to_llm = path_to_folder / llm

    if path_to_llm.exists() is False:
        path_to_llm.mkdir()

    for dataset in datasets:
        path_to_dataset = path_to_llm / dataset

        if path_to_dataset.exists() is False:
            path_to_dataset.mkdir()

        path_to_general_results = path_to_dataset / "final_results.txt"
        if path_to_general_results.exists() is True:
            path_to_general_results.unlink()

        for fold in range(1, 6):
            path_to_fold = path_to_dataset / f"{fold}_fold"

            if path_to_fold.exists() is False:
                path_to_fold.mkdir()

            result_per_word.to_csv(
                path_to_fold / f"results_training_set.csv", index=False
            )
            result_per_word.to_csv(
                path_to_fold / f"results_testing_set.csv", index=False
            )
            correlation_results.to_csv(path_to_fold / "training.csv", index=False)
            correlation_results.to_csv(path_to_fold / "testing.csv", index=False)

            path_to_verbose_results = path_to_fold / "verbose_results.txt"
            if path_to_verbose_results.exists() is True:
                path_to_verbose_results.unlink()
