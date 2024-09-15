import copy
import logging
import math
import typing
from pathlib import Path
from pprint import pprint
import random
import sys
import unicodedata

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
from scipy.stats import spearmanr

from custom_types import ShortUse, Results
from common_extra_processing import create_grouping
from cross_validation import cross_validation_for_dwug_es as cv


random.seed(42)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def get_adj_matrix(
    scores: pd.DataFrame,
    id2int: dict,
    n_sentences,
    fill_diagonal: bool,
    normalize: bool,
    threshold: float = 0.5,
    scaler: typing.Callable = None,
    wic_score: bool = True,
):
    logging.info("building adjacency matrix ...")
    matrix = np.zeros((n_sentences, n_sentences), dtype="float")
    if fill_diagonal is True:
        diagonal_value = 1.0 if wic_score is True or normalize is True else 4.0
        np.fill_diagonal(matrix, diagonal_value)

    if wic_score is True and normalize is False:
        scores["score"] = scaler.transform(scores["score"].to_numpy().reshape(-1, 1))
        threshold = scaler.transform([[threshold]]).item()

    if normalize is True and wic_score is False:
        scores["score"] = get_scaler(scores["score"]).transform(
            scores["score"].to_numpy().reshape(-1, 1)
        )

    for _, row in scores.iterrows():
        x = id2int[ShortUse(row["word"], row["identifier1"])]
        y = id2int[ShortUse(row["word"], row["identifier2"])]

        try:
            if row["score"] >= threshold:
                if wic_score is True:
                    matrix[x, y] = matrix[y, x] = (
                        1 if normalize is True else row["score"]
                    )
                else:
                    matrix[x, y] = matrix[y, x] = row["score"]

        except Exception as e:
            print(row["score"])
            print(scores.word[0])
            print(e)
            sys.exit(1)

    logging.info("adjacency matrix built ...")

    return matrix


def load_data(path: str, wic_data=False):
    logging.info(f"loading data from {path} ...")

    if wic_data is True:
        data = pd.read_csv(f"{path}.scores")
        return data

    data_to_concatenate = []
    for p in Path(path).glob("*scores"):
        word = p.parts[-1].split(".")[1]
        try:
            data = pd.read_json(p)
        except Exception as e:
            print(p.parts[-1], e)

        data["word"] = unicodedata.normalize("NFC", word)

        data_to_concatenate.append(copy.deepcopy(data))

    annotated_data = pd.concat(data_to_concatenate, ignore_index=True)
    mask = annotated_data["score"] == "-"
    filtered_data = annotated_data[~mask]

    filtered_data["score"] = pd.to_numeric(filtered_data["score"], errors="raise")

    logging.info("data loaded ...")

    return filtered_data


def filter_data(data: pd.DataFrame):
    mask1 = data["identifier1"].str.startswith("modern") & data[
        "identifier2"
    ].str.startswith("old")
    mask2 = data["identifier1"].str.startswith("old") & data[
        "identifier2"
    ].str.startswith("modern")
    filter_data = data[mask1 | mask2]

    return filter_data


def compute_jsd(predictions: dict, grouping: pd.DataFrame, method: str = None):
    clusters_to_freq1 = {}
    clusters_to_freq2 = {}

    old_ids_samples = set(grouping[grouping["grouping"] == 1]["ids"].to_list())
    new_ids_samples = set(grouping[grouping["grouping"] == 2]["ids"].to_list())

    if method is None or method != "spectral_clustering":
        for id, cluster in predictions.items():
            if cluster not in clusters_to_freq1:
                clusters_to_freq1[cluster] = 0
            if cluster not in clusters_to_freq2:
                clusters_to_freq2[cluster] = 0

            if id in old_ids_samples:
                clusters_to_freq1[cluster] += 1
            if id in new_ids_samples:
                clusters_to_freq2[cluster] += 1

        c1 = np.array(list(clusters_to_freq1.values()))
        c2 = np.array(list(clusters_to_freq2.values()))
        val = distance.jensenshannon(c1, c2, base=2.0)
        answer = Results(
            jsd=val,
            cluster_to_freq1=clusters_to_freq1,
            cluster_to_freq2=clusters_to_freq2,
        )

    return answer


def get_gold_data(p: str = "../test_data_es.csv"):
    logging.info("  loading gold data ...")
    gold_data = pd.read_csv(p, sep="\t")
    try:
        result = dict(
            zip(
                gold_data["lemma"],
                zip(
                    gold_data["change_graded"],
                    gold_data["change_binary"],
                ),
            )
        )
    except Exception as e:
        result = dict(
            zip(
                gold_data["word"],
                zip(
                    gold_data["change_graded"],
                    gold_data["change_binary"],
                ),
            )
        )

    logging.info("  gold data loaded ...")

    return result


def save_cv_results(results: dict, metadata: dict = None):
    spr_lscd_training = []
    spr_lscd_testing = []

    path_to_save = f"{metadata['path_to_save_results']}/final_results.txt"

    for fold in range(1, 6):
        spr_lscd_training.append(results[fold]["training"]["max_spr_lscd"])
        spr_lscd_testing.append(results[fold]["testing"])

    with open(path_to_save, "a") as f_out:
        f_out.write(f"Avg spr_lscd [train]: {np.array(spr_lscd_training).mean()}\n")
        f_out.write(f"Std spr_lscd [train]: {np.array(spr_lscd_training).std()}\n")
        f_out.write(f"\n")

        f_out.write(f"Avg spr_lscd [testing]: {np.array(spr_lscd_testing).mean()}\n")
        f_out.write(f"Std spr_lscd [testing]: {np.array(spr_lscd_testing).std()}\n")
        f_out.write(f"\n")

        f_out.write(f"best parameters per fold \n")
        for fold in range(1, 6):
            f_out.write(
                f"  Fold {fold} {results[fold]['training']['optimal_parameters']}\n"
            )


def save_results(word: str, result: Results, parameters: dict, path_to_save: str):
    df = pd.read_csv(path_to_save)

    n_rows = df.shape[0]

    df.loc[n_rows, "word"] = word
    df.loc[n_rows, "jsd"] = result.jsd
    df.loc[n_rows, "clusters_to_freq1"] = str(result.cluster_to_freq1)
    df.loc[n_rows, "clusters_to_freq2"] = str(result.cluster_to_freq2)
    df.loc[n_rows, "parameters"] = str(parameters)

    df.to_csv(path_to_save, index=False)


def save_correlation(correlation: float, parameters: dict, path_to_file: str):
    df = pd.read_csv(path_to_file)

    n_rows = df.shape[0]

    df.loc[n_rows, "correlation"] = str(correlation)
    df.loc[n_rows, "parameters"] = str(parameters)

    df.to_csv(path_to_file, index=False)


def calculate_correlation(jsd: dict[str, Results], path_to_gold_data):
    gold_data = get_gold_data(path_to_gold_data)
    pred_change_graded = []
    gold_change_graded = []

    for word in gold_data:
        proccesed_word = unicodedata.normalize("NFC", word)

        try:
            pred_change_graded.append(jsd[proccesed_word].jsd)
            gold_change_graded.append(gold_data[proccesed_word][0])
        except Exception as e:
            logging.warning(f"    {word} is not a tw from the competence ...")

    spr = spearmanr(gold_change_graded, pred_change_graded)[0]
    return spr


def get_scaler(scores: pd.Series):
    return MinMaxScaler().fit(np.array(scores).reshape(-1, 1))


def get_thresholds(scores: pd.Series):
    return [0.5] + list(np.quantile(scores, np.arange(0.1, 1.0, 0.1)))


def get_predictions(
    get_clusters: typing.Callable,
    scores: pd.DataFrame,
    hyperparameter_combinations: typing.List[dict],
    metadata: dict = None,
):
    logging.info("get predictions ...")
    words = scores.word.unique()
    jsd = {}

    kfold = metadata["kfold"]
    name_file = metadata["name_file"]

    for word in words:
        mask = scores["word"] == word
        filtered_scores = scores[mask]

        ids = set(filtered_scores["identifier1"].to_list()).union(
            set(filtered_scores["identifier2"].to_list())
        )

        grouping = pd.DataFrame({"ids": list(ids)})
        grouping["grouping"] = grouping.apply(
            lambda row: 1 if row["ids"].startswith("old") else 2, axis=1
        )

        context = [ShortUse(word=word, id=id) for id in ids]
        n_sentences = len(ids)

        id2int = {value: index for index, value in enumerate(context)}
        adj_matrix = get_adj_matrix(
            filtered_scores,
            id2int,
            n_sentences,
            hyperparameter_combinations["fill_diagonal"],
            hyperparameter_combinations["normalize"],
            threshold=metadata["threshold"] if metadata["wic_data"] is True else -1.0,
            scaler=metadata["scaler"] if metadata["wic_data"] is True else None,
            wic_score=metadata["wic_data"],
        )

        logging.info("calculating predictions ...")
        clusters = get_clusters(
            adj_matrix, hyperparameter_combinations["model_hyperparameters"]
        )
        pred_clusters = {c.id: clusters[id2int[c]] for index, c in enumerate(context)}
        jsd[word] = compute_jsd(pred_clusters, grouping)
        save_results(
            word,
            jsd[word],
            hyperparameter_combinations,
            f"{metadata['path_to_save_results']}/{kfold}_fold/{name_file}.csv",
        )

    logging.info("returning predictions ...")

    return jsd


def get_predictions_without_nclusters(
    get_clusters: typing.Callable,
    scores: pd.DataFrame,
    hyperparameter_combinations: typing.List[dict],
    metadata: dict = None,
):
    logging.info("get predictions without nclusters ...")
    words = scores.word.unique()
    jsd = {}

    kfold = metadata["kfold"]
    name_file = metadata["name_file"]

    for word in words:
        logging.info(f"processing word: {word}")

        mask = scores["word"] == word
        filtered_scores = scores[mask]

        ids = set(filtered_scores["identifier1"].to_list()).union(
            set(filtered_scores["identifier2"].to_list())
        )

        grouping = pd.DataFrame({"ids": list(ids)})
        grouping["grouping"] = grouping.apply(
            lambda row: 1 if row["ids"].startswith("old") else 2, axis=1
        )

        context = [ShortUse(word=word, id=id) for id in ids]
        n_sentences = len(ids)

        id2int = {value: index for index, value in enumerate(context)}
        adj_matrix = get_adj_matrix(
            filtered_scores,
            id2int,
            n_sentences,
            hyperparameter_combinations["fill_diagonal"],
            hyperparameter_combinations["normalize"],
            threshold=metadata["threshold"] if metadata["wic_data"] is True else -1.0,
            scaler=metadata["scaler"] if metadata["wic_data"] is True else None,
            wic_score=metadata["wic_data"],
        )

        logging.info("calculating predictions ...")

        clusters = get_clusters(
            adj_matrix, hyperparameter_combinations["model_hyperparameters"]
        )

        pred_clusters = {c.id: clusters[id2int[c]] for id, c in enumerate(context)}
        jsd[word] = compute_jsd(pred_clusters, grouping)
        save_results(
            word,
            jsd[word],
            hyperparameter_combinations,
            f"{metadata['path_to_save_results']}/{kfold}_fold/{name_file}.csv",
        )

        logging.info("predictions calculated ...")

    logging.info("returning predictions ...")

    return jsd


def eval(
    get_clusters: typing.Callable,
    scores: dict[str, pd.DataFrame],
    test_set: list,
    parameters: dict,
    metadata: dict,
):
    logging.info(f"eval {metadata['method']} method...")

    metadata["name_file"] = "results_testing_set"

    if metadata["wic_data"] is True:
        metadata.pop("scaler", None)
        metadata.pop("threshold", None)

        metadata.update(
            {"scaler": parameters["scaler"], "threshold": parameters["threshold"]}
        )

    for hyperparameters in [parameters]:
        score_filtered = copy.deepcopy(scores[hyperparameters["prompt"]])
        test_scores = score_filtered[score_filtered["word"].isin(test_set)]

        if metadata["method"] in ["ac", "sc"]:
            jsd = get_predictions(
                get_clusters, test_scores, hyperparameters, metadata=metadata
            )
        else:
            jsd = get_predictions_without_nclusters(
                get_clusters, test_scores, hyperparameters, metadata=metadata
            )

        logging.info("  calculating correlation ...")
        spr = calculate_correlation(jsd, metadata["path_to_gold_data"])
        logging.info(" correlation calculated ...")

        logging.info("  saving results ...")
        save_correlation(
            spr,
            hyperparameters,
            f"{metadata['path_to_save_results']}/{metadata['kfold']}_fold/testing.csv",
        )
        logging.info("  results saved ...")

        metadata.pop("scaler", None)
        metadata.pop("threshold", None)

        return spr


def train(
    get_clusters: typing.Callable,
    scores: pd.DataFrame,
    train_set: list,
    hyperparameter_combinations: list,
    metadata: dict = None,
):
    method = metadata["method"]
    optimal_parameters = None
    max_spr_lscd = float("-inf")

    metadata.update({"name_file": "results_training_set"})

    logging.info(f"training {method} method ...")
    number_iterations = len(hyperparameter_combinations)

    for index, hyperparameter in enumerate(hyperparameter_combinations):
        logging.info(f"  {index + 1}/{number_iterations} - {hyperparameter}")

        score_filtered = copy.deepcopy(scores[hyperparameter["prompt"]])
        train_scores = score_filtered[score_filtered["word"].isin(train_set)]

        if metadata["wic_data"] is True:
            thresholds = get_thresholds(train_scores["score"])
            scaler = get_scaler(train_scores["score"])

            metadata.pop("scaler", None)
            metadata.pop("threshold", None)
            metadata.update({"scaler": scaler})
            metadata.update({"threshold": thresholds[hyperparameter["quantile"]]})

        if method in ["ac", "sc"]:
            try:
                jsd = get_predictions(
                    get_clusters, train_scores, hyperparameter, metadata=metadata
                )
            except Exception as e:
                logging.error(f"error processing parameters: {hyperparameter}")
                print(e)

                continue
        else:
            try:
                jsd = get_predictions_without_nclusters(
                    get_clusters, train_scores, hyperparameter, metadata=metadata
                )
            except Exception as e:
                logging.error(f"error processing parameters: {hyperparameter}")
                print(e)
                sys.exit(1)
                continue

        logging.info("  calculating correlation ...")

        spr = calculate_correlation(jsd, metadata["path_to_gold_data"])
        if math.isnan(spr):
            spr = -10.0
        logging.info("  correlation calculated ...")

        logging.info("  saving results ...")
        save_correlation(
            spr,
            hyperparameter,
            f"{metadata['path_to_save_results']}/{metadata['kfold']}_fold/training.csv",
        )
        logging.info("  results saved ...")

        if spr > max_spr_lscd:
            max_spr_lscd = spr
            optimal_parameters = copy.deepcopy(hyperparameter)

            if metadata["wic_data"] is True:
                optimal_parameters["scaler"] = scaler
                optimal_parameters["threshold"] = thresholds[hyperparameter["quantile"]]

    return {"max_spr_lscd": max_spr_lscd, "optimal_parameters": optimal_parameters}


def cross_validation(
    hyperparameter_combinations: list,
    get_clusters: typing.Callable,
    scores: pd.DataFrame,
    metadata: dict = None,
):
    dataset = metadata["dataset"]
    results = {}

    for index in cv[dataset].keys():
        train_set = cv[dataset][index]["train"]
        test_set = cv[dataset][index]["test"]

        metadata.update({"kfold": index})

        configuration = train(
            get_clusters,
            scores,
            train_set,
            hyperparameter_combinations,
            metadata=metadata,
        )

        spr = eval(
            get_clusters,
            scores,
            test_set,
            configuration["optimal_parameters"],
            metadata,
        )

        results[index] = {"training": configuration, "testing": spr}

        path_to_save = (
            f"{metadata['path_to_save_results']}/{index}_fold/verbose_results.txt"
        )
        with open(path_to_save, "a") as f_out:
            f_out.write("best parameters for training: \n")
            f_out.write(f"  {configuration['optimal_parameters']}\n")
            f_out.write(f"training [spr_lscd]: \n")
            f_out.write(f"  {configuration['max_spr_lscd']}\n")

            f_out.write("\n")

            f_out.write(f"testing [spr_lscd]:\n")
            f_out.write(f"  {spr}")

    return results


def grid_search_without_nclusters(
    get_data: typing.Callable,
    get_clusters: typing.Callable,
    hyperparameter_combinations: list,
    metadata: dict = None,
):
    # scores = get_data()
    scores = {}
    for sp in metadata["score_paths"]:
        scores[sp] = load_data(
            f"{metadata['path_to_data']}/{sp}", wic_data=metadata["wic_data"]
        )

    if metadata["dataset"] == "dwug_en":
        create_grouping(scores, metadata["score_paths"], logging)

    results = cross_validation(
        hyperparameter_combinations, get_clusters, scores, metadata=metadata
    )

    save_cv_results(results, metadata=metadata)


def grid_search(
    get_data: typing.Callable,
    get_clusters: typing.Callable,
    hyperparameter_combinations: list,
    metadata: dict = None,
):
    scores = {}
    for sp in metadata["score_paths"]:
        scores[sp] = load_data(
            f"{metadata['path_to_data']}/{sp}", wic_data=metadata["wic_data"]
        )

    if metadata["dataset"] == "dwug_en":
        create_grouping(scores, metadata["score_paths"], logging)

    # scores = get_data()

    results = cross_validation(
        hyperparameter_combinations,
        get_clusters,
        scores,
        metadata=metadata,
    )

    save_cv_results(results, metadata=metadata)


if __name__ == "__main__":
    ...
