import sys
from collections import namedtuple
import typing
import random
import warnings
import logging
from pprint import pprint

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from scipy.sparse.csgraph import laplacian

from cross_validation import cross_validation as cv
from custom_types import Use
from common_generate_results_files import (
    GenerateResultsNoSC,
    GenerateResultsSC,
    Factory,
)
from common_spr import (
    compute_graded_lscd,
    compute_spearman,
    gold_data,
    Results,
)

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

en_en = "xlmr-large..data_train-wic_train-en-en..train_loss-crossentropy_loss..pool-mean..targ_emb-dist_l1ndotn..hs-0..bn-1"
ru_ru = "xlmr-large..data_train-wic_ru-ru..train_loss-crossentropy_loss..pool-mean..targ_emb-dist_l1ndotn..hs-0..bn-1"
rusemshift_train = "xlmr-large..data_train-wic..train_loss-crossentropy_loss..data_ft-rusemshift-data..ft_loss-crossentropy_loss..pool-mean..targ_emb-dist_l1ndotn..hs-0..bn-1..ckpt-nen-nen-weights/train"
rusemshift_finetune = "xlmr-large..data_train-wic..train_loss-crossentropy_loss..data_ft-rusemshift-data..ft_loss-crossentropy_loss..pool-mean..targ_emb-dist_l1ndotn..hs-0..bn-1..ckpt-nen-nen-weights/finetune"
es_es = "WIC_DWUG+XLWSD"
es_es_rss = "WIC+RSS+DWUG+XLWSD"
llama2 = "llama2"
SCORES_PATHS = {
    # "en_en": en_en,
    # "ru_ru": ru_ru,
    # "rusemshift_train": rusemshift_train,
    # "rusemshift_finetune": rusemshift_finetune,
    # "es_es": es_es,
    # "es_es_rss": es_es_rss,
    "llama2": llama2
}
VALIDATION_METHODS = ["silhouette", "calinski", "eigengap"]
random.seed(45678)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


def get_adj_matrix(
    scores: pd.DataFrame,
    id2int: dict,
    n_sentences: int,
    scaler: MinMaxScaler,
    fits_scaler: bool = False,
    threshold: float = 0.5,
    binarize: bool = False,
    fill_diagonal: bool = False,
):
    matrix = np.zeros((n_sentences, n_sentences), dtype="float")
    if fill_diagonal is True:
        np.fill_diagonal(matrix, 1.0)

    if binarize is False and scaler is not None:
        if fits_scaler is True:
            scores["pred_score"] = scaler.fit_transform(
                scores["pred_score"].to_numpy().reshape(-1, 1)
            )
        else:
            scores["pred_score"] = scaler.transform(
                scores["pred_score"].to_numpy().reshape(-1, 1)
            )

        threshold = scaler.transform([[threshold]]).item()

    for _, row in scores.iterrows():
        x = id2int[Use(row.lemma, row.sentence1, row.start1, row.end1)]
        y = id2int[Use(row.lemma, row.sentence2, row.start2, row.end2)]

        if row.pred_score >= threshold:
            matrix[x, y] = matrix[y, x] = 1 if binarize is True else row.pred_score

    return matrix


def save_no_cv_results(results, method):
    train_ari = results["ari"][0]
    parameters_ari = results["ari"][1]
    train_spr_lscd = results["spr_lscd"][0]
    parameters_spr_lscd = results["spr_lscd"][1]

    with open(f"../no-cv-experiments/{method}/final_results.txt", "w") as f_out:
        if method == "spectral_clustering":
            for m in VALIDATION_METHODS:
                f_out.write(f"{m.upper()}:\n")
                f_out.write("  Train\n")
                f_out.write(f"    ari: {train_ari[m]}\n")
                f_out.write("  Parameters\n")
                f_out.write(f"    {parameters_ari[m]}\n\n")

                f_out.write(f"  Train\n")
                f_out.write(f"    spr_lscd: {train_spr_lscd[m]}\n")
                f_out.write("  Parameters\n")
                f_out.write(f"    {parameters_spr_lscd[m]}\n\n")
        else:
            f_out.write("ari\n")
            f_out.write(f"  {train_ari}\n")
            f_out.write(f"parameters\n")
            f_out.write(f"  {parameters_ari}\n\n")

            f_out.write("spr_lscd\n")
            f_out.write(f"  {train_spr_lscd}\n")
            f_out.write(f"parameters\n")
            f_out.write(f"  {parameters_spr_lscd}\n\n")


def save_cv_results(results: dict[str, tuple], method: str):
    train_ari = results["ari"][0]
    test_ari = results["ari"][1]
    parameters_ari = results["ari"][2]
    train_spr_lscd = results["spr_lscd"][0]
    test_spr_lscd = results["spr_lscd"][1]
    parameters_spr_lscd = results["spr_lscd"][2]

    with open(f"../cv-experiments/{method}/final_results.txt", "w") as f_out:
        if method == "spectral_clustering":
            for m in VALIDATION_METHODS:
                f_out.write(f"{m.upper()}:\n")
                dev_ari_ = np.array([train_ari[index][m] for index in range(1, 6)])
                test_ari_ = np.array([test_ari[index][m] for index in range(1, 6)])
                f_out.write("  Train\n")
                f_out.write(f"    Avg ari: {dev_ari_.mean()}\n")
                f_out.write(f"    Std ari: {dev_ari_.std()}\n")

                f_out.write("  Test\n")
                f_out.write(f"    Avg ari: {test_ari_.mean()}\n")
                f_out.write(f"    Std ari: {test_ari_.std()}\n")

                f_out.write("  Parameters ari\n")
                for index in range(1, 6):
                    f_out.write(f"   Fold-{index}: {parameters_ari[index][m]}\n")
                f_out.write("\n")

                dev_spr_lscd_ = np.array(
                    [train_spr_lscd[index][m] for index in range(1, 6)]
                )
                test_spr_lscd_ = np.array(
                    [test_spr_lscd[index][m] for index in range(1, 6)]
                )
                f_out.write("  Train\n")
                f_out.write(f"    Avg spr_lscd: {dev_spr_lscd_.mean()}\n")
                f_out.write(f"    Std spr_lscd: {dev_spr_lscd_.std()}\n")

                f_out.write("  Test\n")
                f_out.write(f"    Avg spr_lscd: {test_spr_lscd_.mean()}\n")
                f_out.write(f"    Std spr_lscd: {test_spr_lscd_.std()}\n")

                f_out.write("  Parameters spr_lscd\n")
                for index in range(1, 6):
                    f_out.write(f"   Fold-{index}: {parameters_spr_lscd[index][m]}\n")
                f_out.write("\n\n")
        else:
            dev_ari_cv = np.array([train_ari[index] for index in range(1, 6)])
            test_ari_cv = np.array([test_ari[index] for index in range(1, 6)])
            f_out.write(f"Avg ari [train]: {dev_ari_cv.mean()}\n")
            f_out.write(f"Std ari [train]: {dev_ari_cv.std()}\n\n")
            f_out.write(f"Avg ari [test]: {test_ari_cv.mean()}\n")
            f_out.write(f"Std ari [test]: {test_ari_cv.std()}\n\n")
            f_out.write("Parameters ari:\n")

            for index in range(5):
                f_out.write(f"  Fold {index+1} ")
                f_out.write(f"{parameters_ari[index+1]}\n")
            f_out.write("\n")

            dev_spr_lscd_ = np.array([train_spr_lscd[index] for index in range(1, 6)])
            test_spr_lscd_ = np.array([test_spr_lscd[index] for index in range(1, 6)])
            f_out.write(f"Avg spr_lscd [train]: {dev_spr_lscd_.mean()}\n")
            f_out.write(f"Std spr_lscd [train]: {dev_spr_lscd_.std()}\n\n")
            f_out.write(f"Avg spr_lscd [test]: {test_spr_lscd_.mean()}\n")
            f_out.write(f"Std spr_lscd [test]: {test_spr_lscd_.std()}\n\n")

            f_out.write("Parameters spr_lscd:\n")

            for index in range(5):
                f_out.write(f"  Fold {index + 1} ")
                f_out.write(f"{parameters_spr_lscd[index+1]}\n")


def load_dwug_sense(dataset_path):
    uses = pd.concat(
        [pd.read_csv(p, sep="\t") for p in Path(dataset_path).glob("data/*/uses.csv")],
        ignore_index=True,
    )
    print(len(uses), "uses loaded")
    senses = pd.concat(
        [
            pd.read_csv(p, sep="\t")
            for p in Path(dataset_path).glob("labels/*/maj_3/labels_senses.csv")
        ],
        ignore_index=True,
    )

    senses = senses[senses.label != "-1"]
    uses = uses.merge(senses, on="identifier", how="inner")
    print(len(uses), "uses left after filtering by annotator agreement")

    gold_data = pd.DataFrame(
        {
            "context_id": uses.identifier,
            "word": uses.lemma,
            "gold_sense_id": uses.label,
            "positions": uses.indexes_target_token,
            "context": uses.context,
            "grouping": uses.grouping,
        }
    )
    gold_data.positions = gold_data.positions.str.replace(":", "-")
    return gold_data


def load_dwug_old_sense(dataset_path):
    dwug_full_data = load_dwug_sense(dataset_path)
    dwug_old_data = dwug_full_data[dwug_full_data["grouping"] == 1]
    return dwug_old_data


def load_dwug_new_sense(dataset_path):
    dwug_full_data = load_dwug_sense(dataset_path)
    dwug_new_data = dwug_full_data[dwug_full_data["grouping"] == 2]
    return dwug_new_data


def eigengapHeuristic(adj_matrix, max_n_clusters):
    L = laplacian(adj_matrix, normed=True)
    n_components = adj_matrix.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1]

    return [x + 1 for x in index_largest_gap if x <= max_n_clusters - 1]


def filter_scored_data(
    scored_sentence_pairs: pd.DataFrame,
    annotated_data: pd.DataFrame,
    debug=True,
):
    if debug is True:
        print(len(scored_sentence_pairs), "scores before filtering")

    keys = set(annotated_data.context + annotated_data.positions)
    scored_sentence_pairs_filtered = scored_sentence_pairs[
        scored_sentence_pairs.apply(
            lambda row: f"{row.sentence1}{row.start1}-{row.end1}", axis=1
        ).isin(keys)
        & scored_sentence_pairs.apply(
            lambda row: f"{row.sentence2}{row.start2}-{row.end2}", axis=1
        ).isin(keys)
    ]
    if debug is True:
        print(len(scored_sentence_pairs_filtered), "scores after filtering")

    return scored_sentence_pairs_filtered


def load_WiC_model_score(wic_scores_root, debug=False):
    sentence_pairs = pd.concat(
        [pd.read_json(path) for path in wic_scores_root.rglob("inputs.data")],
        ignore_index=True,
    )
    if debug is True:
        print(len(sentence_pairs), "input pairs loaded")

    score_paths = [pd.read_json(path) for path in wic_scores_root.rglob("*.scores")]
    if debug is True:
        print(len(score_paths), "files with scores found")

    scored_sentence_pairs = pd.concat(score_paths, ignore_index=True)

    if debug is True:
        print(
            f"loaded {len(scored_sentence_pairs)} scores from {wic_scores_root}\n  scores per example: ",
            scored_sentence_pairs.score.str.len().value_counts(dropna=False).to_dict(),
        )

    # scored_sentence_pairs["pred_score"] = scored_sentence_pairs.score.apply(
    #     lambda s: sum(float(e) for e in s) / len(s)
    # )

    scored_sentence_pairs["pred_score"] = scored_sentence_pairs.score.apply(
        lambda s: int(s)
    )
    sentence_pairs = sentence_pairs.merge(
        scored_sentence_pairs, on="id", how="inner", validate="1:1"
    )

    if debug is True:
        print(len(sentence_pairs), "scores loaded")

    return sentence_pairs


def load_WiC_model_scores(annotated_data: pd.DataFrame):
    score_paths = {}
    wic_root = Path("../DMlong-latestDWUGdeSense-6wics")
    for sp in SCORES_PATHS.keys():
        wic_score = load_WiC_model_score(wic_root / "scores" / f"{SCORES_PATHS[sp]}/")
        score_paths[sp] = filter_scored_data(wic_score, annotated_data)

    return score_paths


def get_scaler(scores: pd.Series):
    return MinMaxScaler().fit(np.array(scores).reshape(-1, 1))


def get_thresholds(scores: pd.Series):
    return [0.5] + list(np.quantile(scores, np.arange(0.1, 1.0, 0.1)))


def generate_hyperparameter_combination_for_cc(
    train_set: list[str], scores_per_model: pd.DataFrame
):
    model_hyperparameter_combinations = {}
    combinations = []

    for wic_model in SCORES_PATHS.keys():
        scores = scores_per_model[wic_model]
        train_scores = scores[scores.lemma.isin(train_set)]
        for number_senses in [10]:
            for attempt in [2000]:
                for iteration in [50000]:
                    combinations.append(
                        {
                            "threshold_cc": get_scaler(train_scores.pred_score)
                            .transform(np.array(0.5).reshape(1, -1))
                            .item(0),
                            "max_attempts": attempt,
                            "max_iters": iteration,
                            "max_senses": number_senses,
                        }
                    )
        model_hyperparameter_combinations[wic_model] = combinations.copy()
        combinations = []

    return model_hyperparameter_combinations


def generate_hyperparameter_combinations(
    model_hyperparameter_combinations: list,
    include_binarize: bool = True,
    word_level_threshold: bool = False,
    fill_diagonal: bool = True,
    percentile: int = 10,
):
    hyperparameter_combinations = []

    for binarize in [True]:  # [True, False] if include_binarize is True else [False]:
        for quantile in range(percentile):
            for fd in [True, False] if fill_diagonal is True else [False]:
                for score_path in SCORES_PATHS.keys():
                    if isinstance(model_hyperparameter_combinations, dict):
                        combinations = model_hyperparameter_combinations[score_path]
                    else:
                        combinations = model_hyperparameter_combinations

                    for model_hyperparameters in combinations:
                        hyperparameter_combinations.append(
                            {
                                "binarize": binarize,
                                "fill_diagonal": fd,
                                "quantile": quantile,
                                "score_path": score_path,
                                "model_hyperparameters": model_hyperparameters,
                            }
                        )

    return hyperparameter_combinations


def get_predictions(
    senses: pd.DataFrame,
    scores: pd.DataFrame,
    binarize: bool,
    fill_diagonal: bool,
    threshold: float,
    scaler: MinMaxScaler,
    model_hyperparameters: dict,
    get_clusters: typing.Callable,
    max_n_clusters: int,
    GenerateResults: GenerateResultsSC,
    extra_information: dict,
):
    words = senses.word.unique()
    ari_silhouette = 0.0
    ari_calinski = 0.0
    ari_eigengap = 0.0

    jsd = {}
    jsd_silhouette = {}
    jsd_calinski = {}
    jsd_eigengap = {}

    for word in words:
        word_senses = senses[senses["word"] == word]
        word_scores = scores[scores["lemma"] == word]

        contexts = list(
            word_senses.apply(
                lambda x: Use(x["word"], x["context"], x["start"], x["end"]),
                axis=1,
            )
        )
        n_sentences = len(contexts)
        id2int = {x: i for i, x in enumerate(contexts)}

        adj_matrix = get_adj_matrix(
            word_scores,
            id2int,
            n_sentences,
            scaler,
            fits_scaler=False,
            threshold=threshold,
            binarize=binarize,
            fill_diagonal=fill_diagonal,
        )

        cluster_rand_score = {x: 0 for x in range(2, max_n_clusters + 1)}
        cluster_silhouette = {x: 0 for x in range(2, max_n_clusters + 1)}
        cluster_calinski = {x: 0 for x in range(2, max_n_clusters + 1)}
        predicted_clusters = {}
        map_id_to_cluster_labels = {}

        eigengap_rand_score = None
        eigengap_n_clusters = eigengapHeuristic(adj_matrix, max_n_clusters)

        for n_clusters in range(1, max_n_clusters + 1):
            clusters = get_clusters(
                adj_matrix,
                n_clusters,
                model_hyperparameters,
                random.randint(1, 10000),
            )
            clusters_and_ids = word_senses.apply(
                lambda x: (
                    clusters[
                        id2int[
                            Use(
                                x["word"],
                                x["context"],
                                x["start"],
                                x["end"],
                            )
                        ]
                    ],
                    x["context_id"],
                ),
                axis=1,
            )
            originalIds2clusterlabel = {
                item[1]: item[0] for item in clusters_and_ids.values
            }
            pred_clusters = clusters_and_ids.apply(lambda obj: obj[0])

            predicted_clusters[n_clusters] = pred_clusters.to_dict()
            map_id_to_cluster_labels[n_clusters] = originalIds2clusterlabel
            rand = metrics.adjusted_rand_score(
                word_senses["gold_sense_id"], pred_clusters
            )

            if n_clusters != 1:
                cluster_rand_score[n_clusters] += rand
                try:
                    rand_silhouette = metrics.silhouette_score(
                        adj_matrix, clusters, metric="euclidean"
                    )
                    rand_calinski = metrics.calinski_harabasz_score(
                        adj_matrix, clusters
                    )
                    cluster_silhouette[n_clusters] += rand_silhouette
                    cluster_calinski[n_clusters] += rand_calinski
                except Exception:
                    rand_silhouette = -1
                    rand_calinski = -1
                    cluster_silhouette[n_clusters] = -1
                    cluster_calinski[n_clusters] = -1

            if eigengap_n_clusters[0] == n_clusters:
                eigengap_rand_score = rand

        ncluster_silhouette = np.array(list(cluster_silhouette.values())).argmax() + 2
        ncluster_calinski = np.array(list(cluster_calinski.values())).argmax() + 2
        param = {
            "ari": max(cluster_rand_score.values()),
            "ari_old": -3.0,
            "ari_silhouette_old": -3.0,
            "ari_calinski_old": -3.0,
            "ari_eigengap_old": -3.0,
            "ari_new": -3.0,
            "ari_silhouette_new": -3.0,
            "ari_calinski_new": -3.0,
            "ari_eigengap_new": -3.0,
            "ari_silhouette": cluster_rand_score[ncluster_silhouette],
            "ari_calinski": cluster_rand_score[ncluster_calinski],
            "ari_eigengap": eigengap_rand_score,
            "number_clusters_selected_by_silhouette": ncluster_silhouette,
            "number_clusters_selected_by_calinski": ncluster_calinski,
            "number_clusters_selected_by_eigengap": eigengap_n_clusters[0],
            "ari_per_ncluster": cluster_rand_score,
        }

        jsd[word] = compute_graded_lscd(
            predicted_clusters, word_senses, "spectral_clustering"
        )
        jsd_silhouette[word] = jsd[word][ncluster_silhouette]
        jsd_calinski[word] = jsd[word][ncluster_calinski]
        jsd_eigengap[word] = jsd[word][eigengap_n_clusters[0]]

        if GenerateResults is not None:
            GenerateResults.add_result(None, param)
            GenerateResults.add_result("wic_model", extra_information["wic_model"])
            GenerateResults.add_result(
                "parameters",
                [
                    {
                        "word": word,
                        "binarize": binarize,
                        "quantile": extra_information["quantile"],
                        "fill_diagonal": fill_diagonal,
                        "wic_model": extra_information["wic_model"],
                        "model_hyperparameters": model_hyperparameters,
                    }
                ],
            )
            GenerateResults.add_result(
                "gold_id", [str(word_senses["gold_sense_id"].to_dict())]
            )
            GenerateResults.add_result("predicted_clusters", [str(predicted_clusters)])
            GenerateResults.add_result(
                "jsd_silhouette", jsd[word][ncluster_silhouette].jsd
            )
            GenerateResults.add_result("jsd_calinski", jsd[word][ncluster_calinski].jsd)
            GenerateResults.add_result(
                "jsd_eigengap", jsd[word][eigengap_n_clusters[0]].jsd
            )
            GenerateResults.add_result(
                "freq_clusters1_silhouette",
                str(jsd[word][ncluster_silhouette].cluster_to_freq1),
            )
            GenerateResults.add_result(
                "freq_clusters2_silhouette",
                str(jsd[word][ncluster_silhouette].cluster_to_freq2),
            )
            GenerateResults.add_result(
                "freq_clusters1_calinski",
                str(jsd[word][ncluster_calinski].cluster_to_freq1),
            )
            GenerateResults.add_result(
                "freq_clusters2_calinski",
                str(jsd[word][ncluster_calinski].cluster_to_freq2),
            )
            GenerateResults.add_result(
                "freq_clusters1_eigengap",
                str(jsd[word][eigengap_n_clusters[0]].cluster_to_freq1),
            )
            GenerateResults.add_result(
                "freq_clusters2_eigengap",
                str(jsd[word][eigengap_n_clusters[0]].cluster_to_freq2),
            )
            GenerateResults.add_result(
                "map_id_to_cluster_label", str(map_id_to_cluster_labels)
            )
            GenerateResults.save(
                f"{extra_information['path_file'].format(kfold=extra_information['kfold'])}/{extra_information['name_file']}"
            )

        ari_silhouette += cluster_rand_score[ncluster_silhouette]
        ari_calinski += cluster_rand_score[ncluster_calinski]
        ari_eigengap += eigengap_rand_score

    return (
        ari_silhouette / len(words),
        ari_calinski / len(words),
        ari_eigengap / len(words),
        {
            "spr_lscd_silhouette": compute_spearman(jsd_silhouette),
            "spr_lscd_calinski": compute_spearman(jsd_calinski),
            "spr_lscd_eigengap": compute_spearman(jsd_eigengap),
        },
    )


def get_predictions_without_nclusters(
    senses: pd.DataFrame,
    scores: pd.DataFrame,
    binarize: bool,
    fill_diagonal: bool,
    threshold: float,
    scaler: MinMaxScaler,
    model_hyperparameters: dict,
    get_clusters: typing.Callable,
    GenerateResults: GenerateResultsNoSC,
    extra_information: dict,
):
    words = senses.word.unique()
    ari = 0.0
    jsd = {}

    for word in words:
        word_senses = senses[senses["word"] == word]
        word_scores = scores[scores["lemma"] == word]

        parameters = {
            "word": word,
            "binarize": binarize,
            "quantile": extra_information["quantile"],
            "fill_diagonal": fill_diagonal,
            "wic_model": extra_information["wic_model"],
            "model_hyperparameters": model_hyperparameters,
        }

        if GenerateResults.is_in_cache(
            f"{extra_information['path_file'].format(kfold=extra_information['kfold'])}/{extra_information['name_file']}",
            parameters=parameters,
            drop_fields=[("model_hyperparameters", "use_disconnected_edges")],
        ):
            ari += GenerateResults.data.loc[GenerateResults.index - 1].ari
            jsd[word] = Results(
                jsd=GenerateResults.data.loc[GenerateResults.index - 1].jsd,
                cluster_to_freq1=None,
                cluster_to_freq2=None,
            )
            continue

        contexts = list(
            word_senses.apply(
                lambda x: Use(x["word"], x["context"], x["start"], x["end"]),
                axis=1,
            )
        )
        n_sentences = len(contexts)
        id2int = {x: i for i, x in enumerate(contexts)}

        adj_matrix = get_adj_matrix(
            word_scores,
            id2int,
            n_sentences,
            scaler,
            fits_scaler=False,
            threshold=threshold,
            binarize=binarize,
            fill_diagonal=fill_diagonal,
        )

        if extra_information["method"] in ["correlation_clustering", "wsbm"]:
            model_hyperparameters["use_disconnected_edges"] = binarize

        clusters = get_clusters(
            adj_matrix, model_hyperparameters, random.randint(1, 10000)
        )
        clusters_and_ids = word_senses.apply(
            lambda x: (
                clusters[
                    id2int[
                        Use(
                            x["word"],
                            x["context"],
                            x["start"],
                            x["end"],
                        )
                    ]
                ],
                x["context_id"],
            ),
            axis=1,
        )
        originalIds2clusterlabel = {
            item[1]: item[0] for item in clusters_and_ids.values
        }
        pred_clusters = clusters_and_ids.apply(lambda obj: obj[0])

        rand = metrics.adjusted_rand_score(word_senses["gold_sense_id"], pred_clusters)

        ari += rand

        jsd[word] = compute_graded_lscd(
            pred_clusters, senses, extra_information["method"]
        )

        if GenerateResults is not None:
            GenerateResults.add_result("ari", rand)
            GenerateResults.add_result(
                "parameters",
                [
                    {
                        "word": word,
                        "binarize": binarize,
                        "quantile": extra_information["quantile"],
                        "fill_diagonal": fill_diagonal,
                        "wic_model": extra_information["wic_model"],
                        "model_hyperparameters": model_hyperparameters,
                    }
                ],
            )
            GenerateResults.add_result("status", "done")
            GenerateResults.add_result(
                "gold_id", str(word_senses["gold_sense_id"].to_dict())
            )
            GenerateResults.add_result(
                "predicted_clusters", str(pred_clusters.to_dict())
            )
            GenerateResults.add_result(
                "number_clusters_predicted",
                len(set(list(pred_clusters.to_dict().values()))),
            )
            # GenerateResults.add_result(
            #     "abs_difference_clusters",
            # )

            GenerateResults.add_result("jsd", jsd[word].jsd)
            GenerateResults.add_result(
                "freq_clusters1", str(jsd[word].cluster_to_freq1)
            )
            GenerateResults.add_result(
                "freq_clusters2", str(jsd[word].cluster_to_freq2)
            )
            GenerateResults.add_result("ari_old", -3.0)
            GenerateResults.add_result("ari_new", -3.0)
            GenerateResults.add_result(
                "map_id_to_cluster_label", str(originalIds2clusterlabel)
            )
            GenerateResults.save(
                f"{extra_information['path_file'].format(kfold=extra_information['kfold'])}/{extra_information['name_file']}"
            )

    return float(ari) / float(len(words)), compute_spearman(jsd)


def train(
    hyperparameter_combinations: dict,
    train_set: list[str],
    senses: pd.DataFrame,
    scores: dict[str, pd.DataFrame],
    get_clusters: typing.Callable,
    method: str,
    max_n_clusters: int = None,
    extra_information: dict = {},
):
    train_senses = senses[senses.word.isin(train_set)].copy()
    best_parameters = None
    best_ari = 0.0
    best_spr_lscd = 0.0
    best_parameters_spr_lscd = None

    best_parameters_silhouette = None
    best_parameters_calinski = None
    best_parameters_eigengap = None
    best_ari_silhouette = 0.0
    best_ari_calinski = 0.0
    best_ari_eigengap = 0.0

    best_parameters_spr_lscd_silhouette = None
    best_parameters_spr_lscd_calinski = None
    best_parameters_spr_lscd_eigengap = None
    best_spr_lscd_silhouette = 0.0
    best_spr_lscd_calinski = 0.0
    best_spr_lscd_eigengap = 0.0

    n = len(hyperparameter_combinations)
    generate_results = Factory(method)
    generate_results.load_cache(
        f"{extra_information['path_file'].format(kfold=extra_information['kfold'])}/{extra_information['name_file']}"
    )
    for index, hyperparameters in enumerate(hyperparameter_combinations):
        print(
            f"Fold {extra_information['kfold']} - Processed {index + 1}/{n} parameters"
        )

        filtered_scores = scores[hyperparameters["score_path"]].copy()
        train_scores = filtered_scores[filtered_scores.lemma.isin(train_set)]

        thresholds = get_thresholds(train_scores.pred_score)
        assert len(thresholds) == 10
        scaler = get_scaler(train_scores.pred_score)
        extra_information["quantile"] = hyperparameters["quantile"]
        extra_information["method"] = method
        extra_information["wic_model"] = hyperparameters["score_path"]

        if method != "spectral_clustering":
            ari, spr_lscd = get_predictions_without_nclusters(
                train_senses,
                train_scores,
                hyperparameters["binarize"],
                hyperparameters["fill_diagonal"],
                thresholds[hyperparameters["quantile"]],
                scaler,
                hyperparameters["model_hyperparameters"],
                get_clusters,
                generate_results,
                extra_information,
            )

            if ari > best_ari:
                best_ari = ari
                best_parameters = hyperparameters.copy()
                best_parameters["scaler"] = scaler
                best_parameters["threshold"] = thresholds[hyperparameters["quantile"]]

            if spr_lscd > best_spr_lscd:
                best_spr_lscd = spr_lscd
                best_parameters_spr_lscd = hyperparameters.copy()
                best_parameters_spr_lscd["scaler"] = scaler
                best_parameters_spr_lscd["threshold"] = thresholds[
                    hyperparameters["quantile"]
                ]

        else:
            assert max_n_clusters is not None, f"max_n_clusters parameter is None"

            (
                ari_silhouette,
                ari_calinski,
                ari_eigengap,
                spr_lscd_predictions,
            ) = get_predictions(
                train_senses,
                train_scores,
                hyperparameters["binarize"],
                hyperparameters["fill_diagonal"],
                thresholds[hyperparameters["quantile"]],
                scaler,
                hyperparameters["model_hyperparameters"],
                get_clusters,
                max_n_clusters,
                generate_results,
                extra_information,
            )

            if ari_silhouette > best_ari_silhouette:
                best_ari_silhouette = ari_silhouette
                best_parameters_silhouette = hyperparameters.copy()
                best_parameters_silhouette["scaler"] = scaler
                best_parameters_silhouette["threshold"] = thresholds[
                    hyperparameters["quantile"]
                ]
            if ari_calinski > best_ari_calinski:
                best_ari_calinski = ari_calinski
                best_parameters_calinski = hyperparameters.copy()
                best_parameters_calinski["scaler"] = scaler
                best_parameters_calinski["threshold"] = thresholds[
                    hyperparameters["quantile"]
                ]
            if ari_eigengap > best_ari_eigengap:
                best_ari_eigengap = ari_eigengap
                best_parameters_eigengap = hyperparameters.copy()
                best_parameters_eigengap["scaler"] = scaler
                best_parameters_eigengap["threshold"] = thresholds[
                    hyperparameters["quantile"]
                ]

            spr_lscd_silhouette = spr_lscd_predictions["spr_lscd_silhouette"]
            spr_lscd_calinski = spr_lscd_predictions["spr_lscd_calinski"]
            spr_lscd_eigengap = spr_lscd_predictions["spr_lscd_eigengap"]

            if spr_lscd_silhouette > best_spr_lscd_silhouette:
                best_spr_lscd_silhouette = spr_lscd_silhouette
                best_parameters_spr_lscd_silhouette = hyperparameters.copy()
                best_parameters_spr_lscd_silhouette["scaler"] = scaler
                best_parameters_spr_lscd_silhouette["threshold"] = thresholds[
                    hyperparameters["quantile"]
                ]
            if spr_lscd_calinski > best_spr_lscd_calinski:
                best_spr_lscd_calinski = spr_lscd_calinski
                best_parameters_spr_lscd_calinski = hyperparameters.copy()
                best_parameters_spr_lscd_calinski["scaler"] = scaler
                best_parameters_spr_lscd_calinski["threshold"] = thresholds[
                    hyperparameters["quantile"]
                ]
            if spr_lscd_eigengap > best_spr_lscd_eigengap:
                best_spr_lscd_eigengap = spr_lscd_eigengap
                best_parameters_spr_lscd_eigengap = hyperparameters.copy()
                best_parameters_spr_lscd_eigengap["scaler"] = scaler
                best_parameters_spr_lscd_eigengap["threshold"] = thresholds[
                    hyperparameters["quantile"]
                ]

    if method != "spectral_clustering":
        return {
            "ari": [best_ari, best_parameters],
            "spr_lscd": [best_spr_lscd, best_parameters_spr_lscd],
        }

    else:
        return {
            "ari": {
                "silhouette": [
                    best_ari_silhouette,
                    best_parameters_silhouette,
                ],
                "calinski": [best_ari_calinski, best_parameters_calinski],
                "eigengap": [best_ari_eigengap, best_parameters_eigengap],
            },
            "spr_lscd": {
                "silhouette": [
                    best_spr_lscd_silhouette,
                    best_parameters_spr_lscd_silhouette,
                ],
                "calinski": [
                    best_spr_lscd_calinski,
                    best_parameters_spr_lscd_calinski,
                ],
                "eigengap": [
                    best_spr_lscd_eigengap,
                    best_parameters_spr_lscd_eigengap,
                ],
            },
        }


def eval(
    hyperparameter_combinations: dict,
    test_set: list[str],
    senses: pd.DataFrame,
    scores: dict[str, pd.DataFrame],
    get_clusters: typing.Callable,
    method: str,
    max_n_clusters: int = None,
    save_results: bool = False,
    extra_information: dict = {},
):
    test_senses = senses[senses.word.isin(test_set)].copy()

    for hyperparameters in [hyperparameter_combinations]:
        filtered_scores = scores[hyperparameters["score_path"]].copy()
        test_scores = filtered_scores[filtered_scores.lemma.isin(test_set)]
        extra_information["method"] = method
        extra_information["quantile"] = hyperparameters["quantile"]
        extra_information["wic_model"] = hyperparameters["score_path"]

        GenerateResults = Factory(method) if save_results is True else None

        if method != "spectral_clustering":
            ari, spr_lscd = get_predictions_without_nclusters(
                test_senses,
                test_scores,
                hyperparameters["binarize"],
                hyperparameters["fill_diagonal"],
                hyperparameters["threshold"],
                hyperparameters["scaler"],
                hyperparameters["model_hyperparameters"],
                get_clusters,
                GenerateResults,
                extra_information,
            )

            return ari, spr_lscd
        else:
            assert max_n_clusters is not None
            (
                ari_silhouette,
                ari_calinski,
                ari_eigengap,
                spr_lscd,
            ) = get_predictions(
                test_senses,
                test_scores,
                hyperparameters["binarize"],
                hyperparameters["fill_diagonal"],
                hyperparameters["threshold"],
                hyperparameters["scaler"],
                hyperparameters["model_hyperparameters"],
                get_clusters,
                max_n_clusters,
                GenerateResults,
                extra_information,
            )

            return {
                "ari": {
                    "silhouette": ari_silhouette,
                    "calinski": ari_calinski,
                    "eigengap": ari_eigengap,
                },
                "spr_lscd": {
                    "silhouette": spr_lscd["spr_lscd_silhouette"],
                    "calinski": spr_lscd["spr_lscd_calinski"],
                    "eigengap": spr_lscd["spr_lscd_eigengap"],
                },
            }


def no_cross_validation(
    hyperparameter_combinations: list,
    senses: pd.DataFrame,
    scores: dict[str, pd.DataFrame],
    get_clusters: typing.Callable,
    method: str,
    max_n_clusters: int = None,
    extra_information: dict = {},
):
    train_set = gold_data.lemma.to_list()
    dev_ari_cv = {}
    parameters_ari_cv = {}

    dev_spr_lscd_cv = {}
    parameters_spr_lscd_cv = {}

    extra_information["kfold"] = "all_words"

    if method == "correlation_clustering":
        hyperparameter_combinations = generate_hyperparameter_combinations(
            generate_hyperparameter_combination_for_cc(train_set, scores)
        )

    results = train(
        hyperparameter_combinations,
        train_set,
        senses,
        scores,
        get_clusters,
        method,
        max_n_clusters,
        extra_information.copy(),
    )

    if method != "spectral_clustering":
        dev_ari_cv, parameters_ari_cv = results["ari"]
        dev_spr_lscd_cv, parameters_spr_lscd_cv = results["spr_lscd"]
    else:
        configuration_ari = results["ari"]
        configuration_spr_lscd = results["spr_lscd"]

        for m in VALIDATION_METHODS:
            dev_ari_cv[m] = configuration_ari[m][0]
            parameters_ari_cv[m] = configuration_ari[m][1]
            dev_spr_lscd_cv[m] = configuration_spr_lscd[m][0]
            parameters_spr_lscd_cv[m] = configuration_spr_lscd[m][1]

    return {
        "ari": [dev_ari_cv, parameters_ari_cv],
        "spr_lscd": [dev_spr_lscd_cv, parameters_spr_lscd_cv],
    }


def cross_validation(
    hyperparameter_combinations: list,
    senses: pd.DataFrame,
    scores: dict[str, pd.DataFrame],
    get_clusters: typing.Callable,
    method: str,
    max_n_clusters: int = None,
    extra_information: dict = {},
):
    dev_ari_cv = {}
    test_ari_cv = {}
    parameters_ari_cv = {}

    dev_spr_lscd_cv = {}
    test_spr_lscd_cv = {}
    parameters_spr_lscd_cv = {}

    for index in cv.keys():
        train_set = cv[index]["train"]
        test_set = cv[index]["test"]
        path_verbose_results = (
            f"{extra_information['path_file'].format(kfold=index)}/verbose_results.txt"
        )

        extra_information.update({"kfold": index})

        if method == "correlation_clustering":
            hyperparameter_combinations = generate_hyperparameter_combinations(
                generate_hyperparameter_combination_for_cc(train_set, scores)
            )

        results = train(
            hyperparameter_combinations,
            train_set,
            senses,
            scores,
            get_clusters,
            method,
            max_n_clusters,
            extra_information.copy(),
        )
        if method != "spectral_clustering":
            train_ari, best_train_parameters = results["ari"]
            train_spr_lscd, best_parameters_spr_lscd = results["spr_lscd"]

            dev_ari_cv[index] = train_ari
            parameters_ari_cv[index] = best_train_parameters
            dev_spr_lscd_cv[index] = train_spr_lscd
            parameters_spr_lscd_cv[index] = best_parameters_spr_lscd
            extra_information["name_file"] = "testing.csv"

            test_ari, test_spr_lscd_exchange_parameters = eval(
                best_train_parameters,
                test_set,
                senses,
                scores,
                get_clusters,
                method,
                None,
                save_results=True,
                extra_information=extra_information.copy(),
            )
            test_ari_exchange_parameters, test_spr_lscd = eval(
                best_parameters_spr_lscd,
                test_set,
                senses,
                scores,
                get_clusters,
                method,
                None,
                save_results=True,
                extra_information=extra_information.copy(),
            )

            with open(
                path_verbose_results,
                "+a",
            ) as f_out:
                f_out.write("training - best parameters for ari\n")
                f_out.write("  " + str(best_train_parameters) + "\n")
                f_out.write("training - [ari]\n")
                f_out.write(f"  {train_ari}\n")
                f_out.write("training - best parameters for spr_lscd\n")
                f_out.write("  " + str(best_parameters_spr_lscd) + "\n")
                f_out.write("traning - [spr_lscd]\n")
                f_out.write(f"  {train_spr_lscd}\n")

                f_out.write("\n")

                f_out.write("testing - [ari]\n")
                f_out.write(f"  {test_ari}\n")
                f_out.write("testing - [spr_lscd]\n")
                f_out.write(f"  {test_spr_lscd}\n")
                f_out.write("\n")

                f_out.write("testing - ari [using best parameters of spr_lscd]\n")
                f_out.write(f"  {test_ari_exchange_parameters}\n")
                f_out.write("\n")

                f_out.write(
                    "testing - spr_lscd [using best parameters of ari]\n",
                )
                f_out.write(f"  {test_spr_lscd_exchange_parameters}\n")

            test_ari_cv[index] = test_ari
            test_spr_lscd_cv[index] = test_spr_lscd
            extra_information["name_file"] = "training.csv"

        else:
            configuration_ari = results["ari"]
            configuration_spr_lscd = results["spr_lscd"]
            extra_information["name_file"] = "testing.csv"

            dev_ari_cv[index] = {}
            test_ari_cv[index] = {}
            parameters_ari_cv[index] = {}
            dev_spr_lscd_cv[index] = {}
            test_spr_lscd_cv[index] = {}
            parameters_spr_lscd_cv[index] = {}
            with open(path_verbose_results, "+a") as f_out:
                for m in VALIDATION_METHODS:
                    dev_ari_cv[index][m] = configuration_ari[m][0]
                    test_ari_cv[index][m] = 0.0
                    parameters_ari_cv[index][m] = configuration_ari[m][1]

                    dev_spr_lscd_cv[index][m] = configuration_spr_lscd[m][0]
                    test_spr_lscd_cv[index][m] = 0.0
                    parameters_spr_lscd_cv[index][m] = configuration_spr_lscd[m][1]

                    f_out.write(f"training - best parameters for ari[{m}]\n")
                    f_out.write(f"  {configuration_ari[m][1]}\n")
                    f_out.write(f"training - ari[{m}]\n")
                    f_out.write(f"  {configuration_ari[m][0]}\n")
                    f_out.write("\n")

                    f_out.write(f"training - best parameters for spr_lscd[{m}]\n")
                    f_out.write(f"  {configuration_spr_lscd[m][1]}\n")
                    f_out.write(f"training - spr_lscd[{m}]\n")
                    f_out.write(f"  {configuration_spr_lscd[m][0]}\n")
                    f_out.write("\n")

                    results_ari = eval(
                        configuration_ari[m][1],
                        test_set,
                        senses,
                        scores,
                        get_clusters,
                        method,
                        max_n_clusters,
                        save_results=True,
                        extra_information=extra_information.copy(),
                    )

                    test_ari_cv[index][m] = results_ari["ari"][m]

                    results_spr_lscd = eval(
                        configuration_spr_lscd[m][1],
                        test_set,
                        senses,
                        scores,
                        get_clusters,
                        method,
                        max_n_clusters,
                        save_results=True,
                        extra_information=extra_information.copy(),
                    )

                    f_out.write(f"testing - ari[{m}]\n")
                    f_out.write(f"  {results_ari['ari'][m]}\n")
                    f_out.write(f"testing - spr_lscd [{m}]\n")
                    f_out.write(f"  {results_spr_lscd['spr_lscd'][m]}\n")
                    f_out.write("\n")

                    f_out.write(
                        f"testing - ari [{m}] [using the best parameters of spr_lscd [{m}]]\n"
                    )
                    f_out.write(f"  {results_spr_lscd['ari'][m]}\n")
                    f_out.write(
                        f"testing - spr_lscd [{m}] [using the best parameters of ari [{m}]]\n"
                    )
                    f_out.write(f"  {results_ari['spr_lscd'][m]}\n")
                    f_out.write("\n")

                    test_spr_lscd_cv[index][m] = results_spr_lscd["spr_lscd"][m]
            extra_information["name_file"] = "training.csv"

    return {
        "ari": [dev_ari_cv, test_ari_cv, parameters_ari_cv],
        "spr_lscd": [
            dev_spr_lscd_cv,
            test_spr_lscd_cv,
            parameters_spr_lscd_cv,
        ],
    }


def grid_search(
    get_data: typing.Callable,
    get_clusters: typing.Callable,
    model_hyperparameter_combinations: list,
    method: str,
    max_number_clusters: int = 5,
    logger_message: dict = None,
    cache: str = None,
    run_experiments: int = 1,
    dataset: str = None,
    is_cross_validation: bool = True,
):
    senses = get_data("../dwug_de_sense/")
    senses = pd.concat(
        [
            senses,
            senses.positions.str.split("-", expand=True)
            .applymap(int)
            .rename(columns={0: "start", 1: "end"}),
        ],
        axis=1,
    )
    scores = load_WiC_model_scores(senses)
    senses.drop(columns="positions", inplace=True)
    hyperparameter_combinations = generate_hyperparameter_combinations(
        model_hyperparameter_combinations,
        include_binarize=True,
        fill_diagonal=True,
        percentile=10,
    )
    extra_information = {
        "dataset": dataset,
        "name_file": "training.csv",
        "path_file": cache,
    }

    if is_cross_validation is True:
        results = cross_validation(
            hyperparameter_combinations,
            senses,
            scores,
            get_clusters,
            method,
            max_number_clusters,
            extra_information,
        )

        save_cv_results(results, method)
    else:
        results = no_cross_validation(
            hyperparameter_combinations,
            senses,
            scores,
            get_clusters,
            method,
            max_n_clusters=5,
            extra_information=extra_information,
        )

        save_no_cv_results(results, method)


def grid_search_without_nclusters(
    get_data: typing.Callable,
    get_clusters: typing.Callable,
    model_hyperparameter_combinations: list,
    include_binarize: bool = False,
    method: str = None,
    logger_message: dict = None,
    cache: str = None,
    run_experiments: int = 1,
    dataset: str = None,
    is_cross_validation: bool = True,
):
    senses = get_data("../dwug_de_sense/")
    senses = pd.concat(
        [
            senses,
            senses.positions.str.split("-", expand=True)
            .applymap(int)
            .rename(columns={0: "start", 1: "end"}),
        ],
        axis=1,
    )
    scores = load_WiC_model_scores(senses)
    senses.drop(columns="positions", inplace=True)
    hyperparameter_combinations = generate_hyperparameter_combinations(
        model_hyperparameter_combinations,
        include_binarize=include_binarize,
        fill_diagonal=True,
        percentile=10,
    )
    extra_information = {
        "dataset": dataset,
        "name_file": "training.csv",
        "path_file": cache,
    }

    if is_cross_validation is True:
        results = cross_validation(
            hyperparameter_combinations,
            senses,
            scores,
            get_clusters,
            method,
            extra_information=extra_information,
        )

        save_cv_results(results, method)
    else:
        results = no_cross_validation(
            hyperparameter_combinations,
            senses,
            scores,
            get_clusters,
            method,
            extra_information=extra_information,
        )

        save_no_cv_results(results, method)


if __name__ == "__main__":
    ...
