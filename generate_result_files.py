from argparse import ArgumentParser
import sys

import pandas as pd
from pathlib import Path
from collections import OrderedDict


methods = [
    "chinese_whispers",
    "correlation_clustering",
    "wsbm",
    "spectral_clustering",
]
datasets = {
    1: "dwug_data_annotated_only",
    2: "dwug_old_data_annotated_only",
    3: "dwug_new_data_annotated_only",
}

parser = ArgumentParser()
parser.add_argument("-p", "--path_file", type=str)
parser.add_argument("-m", "--method", type=str)
# parser.add_argument("-n", "--name", type=str)
parser.add_argument("-d", "--dataset", type=str)
args = parser.parse_args()


file_fields_no_sc = OrderedDict()
file_fields_no_sc["ari"] = []
file_fields_no_sc["parameters"] = []
file_fields_no_sc["status"] = []
file_fields_no_sc["gold_id"] = []
file_fields_no_sc["predicted_clusters"] = []
file_fields_no_sc["number_clusters_predicted"] = []
# file_fields_no_sc["abs_difference_clusters"] = []
file_fields_no_sc["jsd"] = []
file_fields_no_sc["freq_clusters1"] = []
file_fields_no_sc["freq_clusters2"] = []
file_fields_no_sc["ari_old"] = []
file_fields_no_sc["ari_new"] = []
file_fields_no_sc["map_id_to_cluster_label"] = []


file_fields_sc = OrderedDict()
file_fields_sc["ari"] = []
file_fields_sc["ari_old"] = []
file_fields_sc["ari_silhouette_old"] = []
file_fields_sc["ari_calinski_old"] = []
file_fields_sc["ari_eigengap_old"] = []
file_fields_sc["ari_new"] = []
file_fields_sc["ari_silhouette_new"] = []
file_fields_sc["ari_calinski_new"] = []
file_fields_sc["ari_eigengap_new"] = []
file_fields_sc["ari_silhouette"] = []
file_fields_sc["ari_calinski"] = []
file_fields_sc["ari_eigengap"] = []
file_fields_sc["number_clusters_selected_by_silhouette"] = []
file_fields_sc["number_clusters_selected_by_calinski"] = []
file_fields_sc["number_clusters_selected_by_eigengap"] = []
file_fields_sc["ari_per_ncluster"] = []
file_fields_sc["wic_model"] = []
file_fields_sc["parameters"] = []
file_fields_sc["gold_id"] = []
file_fields_sc["predicted_clusters"] = []
file_fields_sc["jsd_silhouette"] = []
file_fields_sc["jsd_calinski"] = []
file_fields_sc["jsd_eigengap"] = []
file_fields_sc["freq_clusters1_silhouette"] = []
file_fields_sc["freq_clusters2_silhouette"] = []
file_fields_sc["freq_clusters1_calinski"] = []
file_fields_sc["freq_clusters2_calinski"] = []
file_fields_sc["freq_clusters1_eigengap"] = []
file_fields_sc["freq_clusters2_eigengap"] = []
file_fields_sc["map_id_to_cluster_label"] = []
# file_fields_sc["abs_difference_clusters_silhouette"] = []
# file_fields_sc["abs_difference_clusters_calinski"] = []
# file_fields_sc["abs_difference_clusters_eigengap"] = []

if args.dataset != "dwug_data_annotated_only":
    del file_fields_no_sc["ari_old"]
    del file_fields_no_sc["ari_new"]

    del file_fields_sc["ari_old"]
    del file_fields_sc["ari_silhouette_old"]
    del file_fields_sc["ari_calinski_old"]
    del file_fields_sc["ari_eigengap_old"]
    del file_fields_sc["ari_new"]
    del file_fields_sc["ari_silhouette_new"]
    del file_fields_sc["ari_calinski_new"]
    del file_fields_sc["ari_eigengap_new"]

if "spectral_clustering" == args.method:
    df = pd.DataFrame(file_fields_sc)
else:
    df = pd.DataFrame(file_fields_no_sc)

df.to_csv(args.path_file, header=True, index=False)
