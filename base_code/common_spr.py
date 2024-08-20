from ast import literal_eval as F
import collections
import math
import sys

import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import spearmanr

from custom_types import Results


gold_data = pd.read_csv("../dwug_de_sense/stats/maj_3/stats_groupings.csv", sep="\t")


def load_gold_lscd_data() -> dict[str, float]:
    data = gold_data[["lemma", "change_graded"]]
    data = data.set_index("lemma")["change_graded"].to_dict()
    return data


def compute_graded_lscd(
    predictions: dict[int, int] | dict[int, dict[int, int]],
    senses: pd.DataFrame,
    method: str,
):
    answer = None
    full_senses = senses.copy()
    old_senses = senses[senses["grouping"] == 1].copy()
    new_senses = senses[senses["grouping"] == 2].copy()

    fulldata_id2context_id = {
        index: full_senses.loc[[index]].context_id.item()
        for index in full_senses.index.to_list()
    }
    old_data_ids = set(old_senses.context_id.to_list())
    new_data_ids = set(new_senses.context_id.to_list())

    cluster_to_freq1 = {}
    cluster_to_freq2 = {}

    if method != "spectral_clustering":
        for id, cluster in predictions.items():
            if cluster not in cluster_to_freq1:
                cluster_to_freq1[cluster] = 0
            if cluster not in cluster_to_freq2:
                cluster_to_freq2[cluster] = 0

            if fulldata_id2context_id[id] in old_data_ids:
                cluster_to_freq1[cluster] += 1
            if fulldata_id2context_id[id] in new_data_ids:
                cluster_to_freq2[cluster] += 1

        c1 = np.array(list(cluster_to_freq1.values()))
        c2 = np.array(list(cluster_to_freq2.values()))
        val = distance.jensenshannon(c1, c2, base=2.0)
        answer = Results(
            jsd=val,
            cluster_to_freq1=cluster_to_freq1,
            cluster_to_freq2=cluster_to_freq2,
        )
    else:
        validation_methods_answer = {}
        for no_clusters, plus_predictions in predictions.items():
            for id, cluster in plus_predictions.items():
                if cluster not in cluster_to_freq1:
                    cluster_to_freq1[cluster] = 0
                if cluster not in cluster_to_freq2:
                    cluster_to_freq2[cluster] = 0

                if fulldata_id2context_id[id] in old_data_ids:
                    cluster_to_freq1[cluster] += 1
                if fulldata_id2context_id[id] in new_data_ids:
                    cluster_to_freq2[cluster] += 1

            c1 = np.array(list(cluster_to_freq1.values()))
            c2 = np.array(list(cluster_to_freq2.values()))
            val = distance.jensenshannon(c1, c2, base=2.0)

            result = Results(
                jsd=val,
                cluster_to_freq1=cluster_to_freq1.copy(),
                cluster_to_freq2=cluster_to_freq2.copy(),
            )
            validation_methods_answer[no_clusters] = result

            cluster_to_freq1.clear()
            cluster_to_freq2.clear()

        answer = validation_methods_answer

    return answer


def compute_spearman(graded_lscd: dict[str, float]):
    gold_lscd_data = load_gold_lscd_data()
    vector1 = [gold_lscd_data[word] for word in graded_lscd.keys()]
    vector2 = [graded_lscd[word].jsd for word in graded_lscd.keys()]
    assert len(vector1) == len(vector2), "lengths of vector icompatible"
    value = spearmanr(vector1, vector2)[0]

    return value


def get_spearman():
    pass
