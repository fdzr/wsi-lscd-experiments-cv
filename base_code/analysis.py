import sys
import logging
from datetime import datetime
from pathlib import Path
from ast import literal_eval as F
from pprint import pprint
import json

import pandas as pd
import numpy as np
from sklearn import metrics
from scipy.stats import spearmanr

from common_spr import compute_spearman, Results
from common_new_functionalities import load_dwug_old_sense, load_dwug_new_sense

# sys.path.append(sys.path[0] + "/..")

logging.basicConfig(format="%(name)s : %(levelname)s : %(message)s", level=logging.INFO)

INPUT_TO_CV_EXPERIMENTS = "../cv-experiments/{method}/{fold}_fold/training.csv"
INPUT_TO_CV_EXPERIMENTS_WSBM = "../cv-experiments-wsbm/wsbm/{fold}/training.csv"
EXPERIMENTS = 4 * [19] + [20]
METHODS = [
    "chinese_whispers",
    "correlation_clustering",
    "wsbm",
    "spectral_clustering",
]


def load_senses_new_old():
    senses_old = load_dwug_old_sense("../dwug_de_sense/")
    senses_new = load_dwug_new_sense("../dwug_de_sense/")
    return senses_old, senses_new


def filter_dataset(
    data: pd.DataFrame,
    field: str,
    filtered_field: str | tuple[str],
    value: str | bool,
):
    if isinstance(filtered_field, tuple):
        data["equal"] = data.apply(
            lambda x: F(x[field])[filtered_field[0]][filtered_field[1]] == value,
            axis=1,
        )
    else:
        data["equal"] = data.apply(
            lambda x: F(x[field])[filtered_field] == value,
            axis=1,
        )
    data = data[data["equal"] == True]
    data.drop(columns="equal", inplace=True)

    return data


def split_dataset(data: pd.DataFrame, chunk: int):
    cant = int(data.shape[0] / chunk)
    for index in range(cant):
        start = index * chunk
        end = start + chunk
        yield data.iloc[start:end, :]


def create_lscd_graded_data(data: pd.DataFrame, field: str):
    pairs = list(
        data.apply(lambda row: (F(row["parameters"])["word"], row[field]), axis=1)
    )

    lscd_graded = {item[0]: Results(item[1], None, None) for item in pairs}
    return lscd_graded


def create_points_to_plot(
    method: str, fold: str, label: str, chunk: int, column: str, jsd_field: str
):
    """
    Create file with points to create scatter plots, ari vs spr_lscd
    """
    data = pd.read_csv(INPUT_TO_CV_EXPERIMENTS.format(method=method, fold=fold))

    # data = filter_dataset(
    #     data.copy(),
    #     "parameters",
    #     ("model_hyperparameters", "distribution"),
    #     "discrete-poisson",
    # )
    # data = filter_dataset(data.copy(), "parameters", "fill_diagonal", False)

    with open("points_for_scatter_plot.txt", "a+") as f_out:
        for subset_data in split_dataset(data, chunk):
            lscd_graded = create_lscd_graded_data(subset_data, jsd_field)
            spr = round(compute_spearman(lscd_graded), 2)
            ari = round(subset_data[f"{column}"].mean(axis=0), 2)

            f_out.write(f"{ari}\t{spr}\t{label}\n")


def create_points():
    labels = {idx: chr(item) for idx, item in enumerate(range(97, 97 + 5), start=1)}
    methods = {"spectral_clustering": "ari_silhouette"}

    for index in range(1, 6):
        for method in methods.keys():
            create_points_to_plot(
                method,
                index,
                labels[index],
                EXPERIMENTS[index - 1],
                "ari_silhouette",
                "jsd_silhouette",
            )


def transform_dataset(senses: pd.DataFrame):
    senses = pd.concat(
        [
            senses,
            senses.positions.str.split("-", expand=True)
            .applymap(int)
            .rename(columns={0: "start", 1: "end"}),
        ],
        axis=1,
    )
    senses.drop(columns="positions", inplace=True)

    return senses


def reorganize_contexts():
    pass


def load_data(path: str):
    data = pd.read_csv(path)
    return data


def save(path: str, results: list):
    with open(path, "a") as f_out:
        for elem in results:
            f_out.write(f"{elem}\n")


def compute_ari(prediction_clusters: list[dict], senses: pd.DataFrame):
    """Compute old ari or new ari depending of the parameters"""

    answer = []

    for pred_clusters in prediction_clusters:
        clusters = F(pred_clusters)
        subset_senses = senses[senses.context_id.isin(list(clusters.keys()))]
        y_pred = [clusters[item] for item in subset_senses.context_id]

        assert len(y_pred) == subset_senses.shape[0], "error matching senses"
        ari = metrics.adjusted_rand_score(subset_senses["gold_sense_id"], y_pred)
        answer.append(ari)

    return answer


def compute_metrics(old_data: bool, cv: bool = True):
    """
    This method is to calculate the old and new ari using the
    full dataset. These results are missing from the main result files.
    """
    if cv is True:
        for fold, index in enumerate(EXPERIMENTS, start=1):
            for m in METHODS:
                p = INPUT_TO_CV_EXPERIMENTS.format(method=m, fold=f"{fold}_fold")
                data = load_data(p)

                for subset_data in split_dataset(data, index):
                    prediction_clusters = subset_data.map_id_to_cluster_label.to_list()
                    res = compute_ari(prediction_clusters, old_data)
                    path_save = "../missing-results/cv-experiments/{method}/{fold}"
                    file_name = "/old_data.txt" if old_data is True else "/new_data.txt"
                    q = path_save.format(method=m, fold=f"{fold}_fold") + file_name
                    save(q, res)


def compute_avg_or_spr_lscd(data: pd.DataFrame, chunk: int, method: str, field: str):
    res = []
    cont = 1
    assert method == "avg" or method == "spr", "invalid method"

    for subset_data in split_dataset(data, chunk):
        logging.info(f"processing subset of word #{cont}")
        if method == "avg":
            value = subset_data[field].mean(axis=0)
        elif method == "spr":
            lscd_data = create_lscd_graded_data(subset_data, field)
            value = compute_spearman(lscd_data)

        res.append(value)
        cont += 1

    return res


def compute_avg_set_of_words():
    res = {}
    start_time = datetime.now()

    for m in ["chinese_whispers"]:
        res[m] = {}
        for index in range(1, 6):
            p = INPUT_TO_CV_EXPERIMENTS.format(method=m, fold=index)
            field_ari = "ari_silhouette" if m == "spectral_clustering" else "ari"
            field_spr = "jsd_silhouette" if m == "spectral_clustering" else "jsd"
            avg_ari = compute_avg_or_spr_lscd(
                pd.read_csv(p), EXPERIMENTS[index - 1], "avg", field_ari
            )
            spr_lscd = compute_avg_or_spr_lscd(
                pd.read_csv(p), EXPERIMENTS[index - 1], "spr", field_spr
            )

            res[m][index] = spearmanr(avg_ari, spr_lscd)[0]
            print(res[m][index])
            print(f"Elapsed time: {datetime.now() - start_time}")
        sys.exit(0)


def explore_nan_values():
    p = INPUT_TO_CV_EXPERIMENTS.format(method="chinese_whispers", fold=1)
    data = pd.read_csv(p)

    for subset_data in split_dataset(data, 19):
        sum = subset_data["jsd"].sum(axis=0)

        if sum == 0.0:
            print()
            for item in zip(subset_data.freq_clusters1, subset_data.freq_clusters2):
                print(item[0], item[1])
            pprint(subset_data.parameters.to_list()[0])
            pprint(subset_data.jsd.to_list())
            sys.exit(0)


def find_rows_by_parameters(data: pd.DataFrame, chunk: int, parameter: dict):
    pass


def load_gold_data():
    gold_data = pd.read_csv(
        "../dwug_de_sense/stats/maj_3/stats_groupings.csv", sep="\t"
    )
    lemmas = gold_data["lemma"].to_list()
    gold_cluster_freq_dists = gold_data["cluster_freq_dist"].to_list()
    gold_cluster_freq_dists = [json.loads(dist) for dist in gold_cluster_freq_dists]
    gold_jsd = gold_data["change_graded"].to_list()
    lemma2gold_cluster_freq_dist = {
        lemmas[i]: gold_cluster_freq_dists[i] for i in range(len(lemmas))
    }
    lemma2gold_jsd = {lemmas[i]: gold_jsd[i] for i in range(len(lemmas))}

    return lemma2gold_jsd


def load_data_v2(partition: str):
    assert (
        partition == "training" or partition == "testing"
    ), "error, it's traning or testing"

    input_path = "../cv-experiments/"
    results = pd.DataFrame()

    for p in Path(input_path).rglob(f"{partition}.csv"):
        path_in_parts = p.parts

        fold = path_in_parts[-2].split("_")[0]
        method = path_in_parts[2]

        data = pd.read_csv(p)
        logging.info(f"Loading {method} results")

        data["method"] = method
        data["fold"] = fold
        data["word"] = data.parameters.map(lambda row: F(row)["word"])
        results = pd.concat([results, data])

    return results


def process_sc_data(data: pd.DataFrame, validation_method: str = "silhouette"):
    new_data = data.copy()
    new_data.loc[new_data.method.eq("spectral_clustering"), "ari"] = new_data[
        new_data.method.eq("spectral_clustering")
    ][f"ari_{validation_method}"]
    new_data.loc[
        new_data.method.eq("spectral_clustering"), "number_clusters_predicted"
    ] = new_data[new_data.method.eq("spectral_clustering")][
        f"number_clusters_selected_by_{validation_method}"
    ]
    new_data.loc[new_data.method.eq("spectral_clustering"), "jsd"] = new_data[
        new_data.method.eq("spectral_clustering")
    ][f"jsd_{validation_method}"]

    return new_data


def calculate_abs_difference(data: pd.DataFrame):
    new_data = data.copy()
    new_data["abs_difference"] = new_data.apply(
        lambda row: abs(
            len(set(F(row["gold_id"]).values())) - row["number_clusters_predicted"]
        ),
        axis=1,
    )

    return new_data


def remove_word_from_parameters(data: pd.DataFrame):
    def remove_word(parameters: str):
        p = F(parameters)
        p.pop("word", None)
        return str(p)

    new_data = data.copy()
    new_data["model"] = new_data.parameters.map(lambda row: remove_word(row))
    return new_data


def find_subset_based_on_best_metrics(data: pd.DataFrame, lemma2gold_jsd: dict):
    results = {}

    data_gb_method = data.groupby("method")
    for method in data_gb_method.groups.keys():
        data_gb_fold = data_gb_method.get_group(method).groupby("fold")
        results[method] = {}

        for fold in data_gb_fold.groups.keys():
            data_gb_model = data_gb_fold.get_group(fold).groupby("model")
            results[method][int(fold)] = {"ari": {}, "spr_lscd": {}}

            best_ari = 0.0
            best_spr_lscd = -2.0
            best_subset_ari = None
            best_subset_spr_lscd = None

            for model in data_gb_model.groups.keys():
                data_filtered = data_gb_model.get_group(model)

                if (ari := data_filtered.ari.mean(axis=0)) > best_ari:
                    best_ari = ari
                    best_subset_ari = data_filtered.copy()

                jsd_predicted = data_filtered["jsd"].to_list()
                jsd_gold = [
                    lemma2gold_jsd[word] for word in data_filtered["word"].to_list()
                ]
                assert len(jsd_gold) == len(jsd_predicted)

                spr_lscd, _ = spearmanr(jsd_gold, jsd_predicted)
                if spr_lscd > best_spr_lscd:
                    best_spr_lscd = spr_lscd
                    best_subset_spr_lscd = data_filtered.copy()

            results[method][int(fold)]["ari"] = best_subset_ari.copy()
            results[method][int(fold)]["spr_lscd"] = best_subset_spr_lscd.copy()

    return results


def compare_abs_difference_training_set(data: dict):
    for m in METHODS:
        abs_difference_ari = 0.0
        abs_difference_spr_lscd = 0.0

        for fold in range(1, 6):
            abs_difference_ari += data[m][fold]["ari"].abs_difference.mean(axis=0)
            abs_difference_spr_lscd += data[m][fold]["spr_lscd"].abs_difference.mean(
                axis=0
            )
            # if m == "spectral_clustering":
            #     print(f"{data[m][fold]['spr_lscd'].model.to_list()}")
            #     print()

        print(f"{m}", sep=" ")
        print(f"abs_difference[ari]: {round(abs_difference_ari / 5, 3)}", sep=" ")
        print(f"abs_difference[spr_lscd]: {round(abs_difference_spr_lscd / 5, 3)}")
        print()


def test(data: pd.DataFrame):
    pass


def compare_ari_old_and_new(partition: str, validation_method: str = "silhouette"):
    data = load_data_v2(partition)
    data = process_sc_data(data, validation_method)
    data = calculate_abs_difference(data)
    data = remove_word_from_parameters(data)
    senses_old, senses_new = load_senses_new_old()
    senses_old = transform_dataset(senses_old)
    senses_new = transform_dataset(senses_new)

    def get_ari_old_and_new(
        row: pd.Series, senses: pd.DataFrame, validation_method: str = "silhouette"
    ):
        if row["method"] == "spectral_clustering":
            map_id_to_clusters = F(row["map_id_to_cluster_label"])[
                int(row[f"number_clusters_selected_by_{validation_method}"])
            ]
            ari = compute_ari([str(map_id_to_clusters)], senses)
        else:
            ari = compute_ari([row["map_id_to_cluster_label"]], senses)

        assert len(ari) == 1

        return ari[0]

    data["ari_old"] = data.apply(
        lambda row: get_ari_old_and_new(row, senses_old, validation_method), axis=1
    )
    data["ari_new"] = data.apply(
        lambda row: get_ari_old_and_new(row, senses_new, validation_method), axis=1
    )

    data_gb_method = data.groupby("method")
    for method in data_gb_method.groups.keys():
        data_gb_fold = data_gb_method.get_group(method).groupby("fold")

        ari_old, ari_new = 0.0, 0.0
        ari_old_list, ari_new_list = [], []

        for fold in data_gb_fold.groups.keys():
            data_filtered = data_gb_fold.get_group(fold)
            ndim = data_filtered.shape[0]

            if method != "spectral_clustering":
                data_filtered_ = data_filtered[: int(ndim / 2)]
                assert int(data_filtered_.shape[0]) == int(ndim / 2)
            else:
                cant_elems_per_sets = int(ndim / 3)
                data_filtered_ = data_filtered.iloc[
                    0 * cant_elems_per_sets : 0 * cant_elems_per_sets
                    + cant_elems_per_sets
                ]

                assert cant_elems_per_sets == data_filtered_.shape[0]
                data_filtered_ = data_filtered_[: int(cant_elems_per_sets / 2)]
                assert int(data_filtered_.shape[0]) == int(cant_elems_per_sets / 2)

            ari_old += data_filtered_.ari_old.mean(axis=0)
            ari_new += data_filtered_.ari_new.mean(axis=0)
            ari_old_list.append(data_filtered_.ari_old.mean(axis=0))
            ari_new_list.append(data_filtered_.ari_new.mean(axis=0))

        print(f"{method}")
        print(
            f"  ari_old: {round(ari_old / 5, 3)} std: {np.std(ari_old_list)}", sep=" "
        )
        print(f"  ari_new: {round(ari_new / 5, 3)} std: {np.std(ari_new_list)}")
        print()


if __name__ == "__main__":
    start_time = datetime.now()
    data = load_data_v2("training")
    data = process_sc_data(data, "eigengap")
    data = calculate_abs_difference(data)
    data = remove_word_from_parameters(data)
    gold_data = load_gold_data()
    results = find_subset_based_on_best_metrics(data, gold_data)
    compare_abs_difference_training_set(results)
    # compare_ari_old_and_new("testing", "silhouette")
    print(f"Elapsed time: {datetime.now() - start_time}")
