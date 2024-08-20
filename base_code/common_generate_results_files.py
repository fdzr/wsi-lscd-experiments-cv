from ast import literal_eval as F
import sys
import pandas as pd


class CacheMixin:
    n_rows = None
    data = None
    jsd = None
    ari = None
    parameters = None

    def is_in_cache(
        self,
        path: str,
        reset_cache: bool = False,
        parameters: dict = None,
        drop_fields: list = [],
    ):
        if reset_cache is True:
            self.reset_cache()

        if self.data is None:
            self.load_cache(path)

        self.index = 0
        for result in self.data.parameters.to_list():
            cache_parameters = F(result)
            self.index += 1
            for field in drop_fields:
                if isinstance(field, str):
                    del cache_parameters[field]
                elif isinstance(field, tuple):
                    del cache_parameters[field[0]][field[1]]

            if cache_parameters == parameters:
                return True

        return False

    def load_cache(self, path):
        self.data = pd.read_csv(path)

    def reset_cache(self):
        self.data = None


class ResultsSCMixin:
    def add_result(self, field=None, value=None):
        if field is None:
            self.df = pd.DataFrame([value])
        else:
            self.df[field] = value

    def save(self, path_file):
        self.df.to_csv(path_file, mode="a", header=False, index=False)


class ResultsNoSCMixin:
    fields = {}
    cache = None
    n_row = None

    def add_result(self, field, value):
        self.n_row = self.df.shape[0]
        self.df.loc[self.n_row - 1, field] = value

    def save(self, path_file):
        self.df.to_csv(path_file, mode="a", header=False, index=False)


class GenerateResultsNoSC(ResultsNoSCMixin, CacheMixin):
    def __init__(self):
        self.initial_fields = {
            "ari": [-1.0],
            "parameters": [None],
            "status": ["processing"],
            "gold_id": [None],
            "predicted_clusters": [None],
        }
        self.df: pd.DataFrame = pd.DataFrame(self.initial_fields)


class GenerateResultsSC(ResultsSCMixin):
    def __init__(self):
        self.df: pd.DataFrame = None


def Factory(method):
    if method in ["chinese_whispers", "correlation_clustering", "wsbm"]:
        return GenerateResultsNoSC()

    return GenerateResultsSC()


if __name__ == "__main__":
    obj = Factory("wsbm")
    ans = obj.is_in_cache(
        "./cv-experiments-wsbm/wsbm/1_fold/training.csv",
        reset_cache=False,
        parameters={
            "wic_model": "es_es_rss",
            "word": "Manschette",
            "binarize": False,
            "quantile": 5,
            "fill_diagonal": False,
            "model_hyperparameters": {"distribution": "real-exponential"},
        },
        drop_fields=[("model_hyperparameters", "use_disconnected_edges")],
    )

    print(ans)
    print(obj.data.loc[obj.index - 1].ari)
