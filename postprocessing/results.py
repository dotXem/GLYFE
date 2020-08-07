import pandas as pd
import os
import numpy as np
from postprocessing.metrics import *
import misc.datasets
import misc.constants as cs
from pathlib import Path
import misc.datasets
from misc.utils import print_latex
from abc import ABC, abstractmethod


class _Results(ABC):
    def __init__(self, model, experiment, ph, dataset):
        self.model = model
        self.experiment = experiment
        self.ph = ph
        self.dataset = dataset
        self.freq = np.max([misc.constants.freq, misc.datasets.datasets[dataset]["glucose_freq"]])

    @abstractmethod
    def compute_results(self, details=False):
        return None, None

    def to_latex(self, table="general"):
        """
        Format the results into a string for the paper in LATEX
        :param table: either "acc" or "cg_ega", corresponds to the table
        :param model_name: prefix of the string, name of the model
        :return:
        """
        mean, std = self.compute_results()
        if table == "p_ega":
            keys = ["P_EGA_A+B", "P_EGA_A", "P_EGA_B", "P_EGA_C", "P_EGA_D", "P_EGA_E"]
        elif table == "r_ega":
            keys = ["R_EGA_A+B", "R_EGA_A", "R_EGA_B", "R_EGA_lC", "R_EGA_uC", "R_EGA_lD", "R_EGA_uD", "R_EGA_lE",
                    "R_EGA_uC"]
        elif table == "cg_ega":
            keys = ["CG_EGA_AP_hypo", "CG_EGA_BE_hypo", "CG_EGA_EP_hypo", "CG_EGA_AP_eu", "CG_EGA_BE_eu",
                    "CG_EGA_EP_eu", "CG_EGA_AP_hyper", "CG_EGA_BE_hyper", "CG_EGA_EP_hyper"]
        elif table == "general":
            keys = ["RMSE", "MAPE", "TG", "CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"]
        elif table == "acc":
            keys = ["RMSE", "MAPE"]
        else:
            keys = ["RMSE"]

        mean = [mean[k] * 100 if "EGA" in k else mean[k] for k in keys]
        std = [std[k] * 100 if "EGA" in k else std[k] for k in keys]

        print_latex(mean, std, label=self.model)


class ResultsDataset(_Results):
    def __init__(self, model, experiment, ph, dataset):
        """
        Object that compute all the performances of a given dataset for a given model and experiment and prediction horizon
        :param model: name of the model (e.g., "base")
        :param experiment: name of the experiment (e.g., "test")
        :param ph: prediction horizons in minutes (e.g., 30)
        :param dataset: name of the dataset (e.g., "ohio")
        """

        super().__init__(model, experiment, ph, dataset)
        self.subjects = misc.datasets.datasets[self.dataset]["subjects"]

    def compute_results(self, details=False):
        """
        Loop through the subjects of the dataset, and compute the mean performances
        :return: mean of metrics, std of metrics
        """
        res = []
        for subject in self.subjects:
            res_subject = ResultsSubject(self.model, self.experiment, self.ph, self.dataset, subject).compute_results()

            if details:
                print(self.dataset, subject, res_subject)

            res.append(res_subject[0])

        keys = list(res[0].keys())
        res = [list(res_.values()) for res_ in res]
        mean, std = np.nanmean(res, axis=0), np.nanstd(res, axis=0)
        return dict(zip(keys, mean)), dict(zip(keys, std))

    def compute_average_params(self):
        params = []
        for subject in self.subjects:
            res_subject = ResultsSubject(self.model, self.experiment, self.ph, self.dataset, subject)
            params.append(res_subject.params)

        return dict(zip(params[0].keys(), np.mean([list(_.values()) for _ in params], axis=0)))


class ResultsSubject(_Results):
    def __init__(self, model, experiment, ph, dataset, subject, params=None, results=None):
        """
        Object that compute all the performances of a given subject for a given model and experiment and prediction horizon
        :param model: name of the model (e.g., "base")
        :param experiment: name of the experiment (e.g., "test")
        :param ph: prediction horizons in minutes (e.g., 30)
        :param dataset: name of the dataset (e.g., "ohio")
        :param subject: name of the subject (e.g., "559")
        :param params: if params and results  are given, performances are directly compute on them, and both are saved into a file
        :param results: see params
        """
        super().__init__(model, experiment, ph, dataset)
        self.subject = subject

        if results is None and params is None:
            self.params, self.results = self.load_raw_results()
        else:
            self.results = results
            self.params = params
            self.save_raw_results()

    def load_raw_results(self):
        """
        Load the results from previous instance of ResultsSubject that has saved the them
        :return: (params dictionary), dataframe with ground truths and predictions
        """
        file = self.dataset + "_" + self.subject + ".npy"
        path = os.path.join(cs.path, "results", self.model, self.experiment, "ph-" + str(self.ph), file)

        params, results = np.load(path, allow_pickle=True)
        dfs = []
        for result in results:
            df = pd.DataFrame(result, columns=["datetime", "y_true", "y_pred"])
            df = df.set_index("datetime")
            df = df.astype("float32")
            dfs.append(df)
        return params, dfs

    def save_raw_results(self):
        """
        Save the results and params
        :return:
        """
        dir = os.path.join(cs.path, "results", self.model, self.experiment, "ph-" + str(self.ph))
        Path(dir).mkdir(parents=True, exist_ok=True)

        saveable_results = np.array([res.reset_index().to_numpy() for res in self.results])

        np.save(os.path.join(dir, self.dataset + "_" + self.subject + ".npy"), [self.params, saveable_results])

    def _compute_raw_results(self, split_by_day=False):
        """
        Compute the raw metrics results for every split (or day, if split_by_day)
        :param split_by_day: wether the results are computed first by day and averaged, or averaged globally
        :return: dictionary of arrays of scores for the metrics
        """
        if split_by_day:
            results = []
            for res in self.results:
                for group in res.groupby(res.index.day):
                    results.append(group[1])
        else:
            results = self.results

        rmse_score = [rmse.RMSE(res_day) for res_day in results]
        mape_score = [mape.MAPE(res_day) for res_day in results]
        mase_score = [mase.MASE(res_day, self.ph, self.freq) for res_day in results]
        tg_score = [time_lag.time_gain(res_day, self.ph, self.freq, "mse") for res_day in results]
        cg_ega_score = np.array([cg_ega.CG_EGA(res_day, self.freq).simplified() for res_day in results])
        cg_ega_score2 = np.array([cg_ega.CG_EGA(res_day, self.freq).reduced() for res_day in results])
        p_ega_score = np.array([p_ega.P_EGA(res_day, self.freq).mean() for res_day in results])
        p_ega_a_plus_b_score = [p_ega.P_EGA(res_day, self.freq).a_plus_b() for res_day in results]
        r_ega_score = np.array([r_ega.R_EGA(res_day, self.freq).mean() for res_day in results])
        r_ega_a_plus_b_score = [r_ega.R_EGA(res_day, self.freq).a_plus_b() for res_day in results]

        return {
            "RMSE": rmse_score,
            "MAPE": mape_score,
            "MASE": mase_score,
            "TG": tg_score,
            "CG_EGA_AP": cg_ega_score2[:, 0],
            "CG_EGA_BE": cg_ega_score2[:, 1],
            "CG_EGA_EP": cg_ega_score2[:, 2],
            "CG_EGA_AP_hypo": cg_ega_score[:, 0],
            "CG_EGA_BE_hypo": cg_ega_score[:, 1],
            "CG_EGA_EP_hypo": cg_ega_score[:, 2],
            "CG_EGA_AP_eu": cg_ega_score[:, 3],
            "CG_EGA_BE_eu": cg_ega_score[:, 4],
            "CG_EGA_EP_eu": cg_ega_score[:, 5],
            "CG_EGA_AP_hyper": cg_ega_score[:, 6],
            "CG_EGA_BE_hyper": cg_ega_score[:, 7],
            "CG_EGA_EP_hyper": cg_ega_score[:, 8],
            "P_EGA_A+B": p_ega_a_plus_b_score,
            "P_EGA_A": p_ega_score[:, 0],
            "P_EGA_B": p_ega_score[:, 1],
            "P_EGA_C": p_ega_score[:, 2],
            "P_EGA_D": p_ega_score[:, 3],
            "P_EGA_E": p_ega_score[:, 4],
            "R_EGA_A+B": r_ega_a_plus_b_score,
            "R_EGA_A": r_ega_score[:, 0],
            "R_EGA_B": r_ega_score[:, 1],
            "R_EGA_uC": r_ega_score[:, 2],
            "R_EGA_lC": r_ega_score[:, 3],
            "R_EGA_uD": r_ega_score[:, 4],
            "R_EGA_lD": r_ega_score[:, 5],
            "R_EGA_uE": r_ega_score[:, 6],
            "R_EGA_lE": r_ega_score[:, 7],
        }

    def compute_results(self, split_by_day=False):
        """
        From the raw metrics scores, compute the mean and std
        :param split_by_day: wether the results are computed first by day and averaged, or averaged globally
        :return: mean of dictionary of metrics, std of dictionary of metrics
        """
        raw_results = self._compute_raw_results(split_by_day=split_by_day)

        mean = {key: val for key, val in zip(list(raw_results.keys()), np.nanmean(list(raw_results.values()), axis=1))}
        std = {key: val for key, val in zip(list(raw_results.keys()), np.nanstd(list(raw_results.values()), axis=1))}

        return mean, std

    def plot(self, day_number=0):
        """
        Plot a given day
        :param day_number: day (int) to plot
        :return: /
        """
        cg_ega.CG_EGA(self.results[0], self.freq).plot(day_number)
