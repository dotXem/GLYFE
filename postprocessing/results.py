import pandas as pd
import os
import numpy as np
from postprocessing.metrics import *
import misc.datasets
import misc.constants as cs
from pathlib import Path
import misc.datasets


class ResultsDataset():
    def __init__(self, model, experiment, ph, dataset, legacy=False):
        """
        Object that compute all the performances of a given dataset for a given model and experiment and prediction horizon
        :param model: name of the model (e.g., "base")
        :param experiment: name of the experiment (e.g., "test")
        :param ph: prediction horizons in minutes (e.g., 30)
        :param dataset: name of the dataset (e.g., "ohio")
        :param legacy: used for old results without the params field in them #TODO remove
        """

        self.model = model
        self.experiment = experiment
        self.ph = ph
        self.dataset = dataset
        self.freq = misc.constants.freq
        self.legacy = legacy
        self.subjects = misc.datasets.datasets[self.dataset]["subjects"]

    def compute_results(self):
        """
        Loop through the subjects of the dataset, and compute the mean performances
        :return: mean of metrics, std of metrics
        """
        res = []
        for subject in self.subjects:
            res_subject = ResultsSubject(self.model, self.experiment, self.ph, self.dataset, subject,
                                         legacy=self.legacy)
            res.append(res_subject.compute_results())

        keys = list(res[0].keys())
        res = [list(res_.values()) for res_ in res]
        mean, std = np.nanmean(res, axis=0), np.nanstd(res, axis=0)
        return dict(zip(keys, mean)), dict(zip(keys, std))

    def to_latex(self, table="acc", model_name=None):
        """
        Format the results into a string for the paper in LATEX
        :param table: either "acc" or "cg_ega", corresponds to the table
        :param model_name: prefix of the string, name of the model
        :return:
        """
        mean, std = self.compute_results()
        # res = np.c_[mean, std]
        name = model_name if model_name is not None else self.model + "_" + self.experiment
        if table == "cg_ega":
            cg_ega_keys = ["CG_EGA_AP_hypo", "CG_EGA_BE_hypo", "CG_EGA_EP_hypo", "CG_EGA_AP_eu", "CG_EGA_BE_eu",
                           "CG_EGA_EP_eu", "CG_EGA_AP_hyper", "CG_EGA_BE_hyper", "CG_EGA_EP_hyper"]
            mean = [mean[k] for k in cg_ega_keys]
            std = [std[k] for k in cg_ega_keys]
        elif table == "acc":
            acc_keys = ["RMSE", "MAPE", "TG"]
            mean = [mean[k] for k in acc_keys]
            std = [std[k] for k in acc_keys]

        str = "\\textbf{" + name + "} & " + " & ".join(
            ["{0:.2f} $\pm$ {1:.2f}".format(mean_, std_) for mean_, std_ in zip(mean, std)]) + "\\\\"
        return str


class ResultsSubject():
    def __init__(self, model, experiment, ph, dataset, subject, params=None, results=None, legacy=False):
        """
        Object that compute all the performances of a given subject for a given model and experiment and prediction horizon
        :param model: name of the model (e.g., "base")
        :param experiment: name of the experiment (e.g., "test")
        :param ph: prediction horizons in minutes (e.g., 30)
        :param dataset: name of the dataset (e.g., "ohio")
        :param subject: name of the subject (e.g., "559")
        :param params: if params and results  are given, performances are directly compute on them, and both are saved into a file
        :param results: see params
        :param legacy: used for old results without the params field in them #TODO remove
        """
        self.model = model
        self.experiment = experiment
        self.ph = ph
        self.dataset = dataset
        self.subject = subject
        self.freq = misc.constants.freq

        if results is None and params is None:
            if not legacy:
                self.params, self.results = self.load_raw_results(legacy)
            else:
                self.results = self.load_raw_results(legacy)
        else:
            self.results = results
            self.params = params
            self.save_raw_results()

        # self.results = self._format_results(self.results)

    def load_raw_results(self, legacy=False):
        """
        Load the results from previous instance of ResultsSubject that has saved the them
        :param legacy: if legacy object shall  be used
        :return: (params dictionary), dataframe with ground truths and predictions
        """
        file = self.dataset + "_" + self.subject + ".npy"
        path = os.path.join(cs.path, "results", self.model, self.experiment, "ph-" + str(self.ph), file)

        if not legacy:
            params, results = np.load(path, allow_pickle=True)
            dfs = []
            for result in results:
                df = pd.DataFrame(result, columns=["datetime", "y_true", "y_pred"])
                df = df.set_index("datetime")
                df = df.astype("float32")
                dfs.append(df)
            return params, dfs
        else:
            results = np.load(path, allow_pickle=True)
            dfs = []
            for result in results:
                df = pd.DataFrame(result, columns=["datetime", "y_true", "y_pred"])
                df = df.set_index("datetime")
                df = df.astype("float32")
                dfs.append(df)
            return dfs

    def save_raw_results(self):
        """
        Save the results and params
        :return:
        """
        dir = os.path.join(cs.path, "results", self.model, self.experiment, "ph-" + str(self.ph))
        Path(dir).mkdir(parents=True, exist_ok=True)

        saveable_results = np.array([res.reset_index().to_numpy() for res in self.results])

        np.save(os.path.join(dir, self.dataset + "_" + self.subject + ".npy"), [self.params, saveable_results])
        # np.save(os.path.join(dir, self.dataset + "_" + self.subject + ".npy"), np.array())

    def compute_results(self):
        """
        Compute the results by averaging on the whole days (that can be not continuous, because of cross-validation)
        :return: dictionary with performances per metric
        """
        rmse_score = np.mean([rmse.RMSE(res_day) for res_day in self.results])
        mape_score = np.mean([mape.MAPE(res_day) for res_day in self.results])
        mase_score = np.mean([mase.MASE(res_day, self.ph, self.freq) for res_day in self.results])
        tg_score = np.mean([time_lag.time_gain(res_day, self.ph, self.freq) for res_day in self.results])
        cg_ega_score = np.mean([cg_ega.CG_EGA(res_day, self.freq).simplified() for res_day in self.results], axis=0)

        results = {
            "RMSE": rmse_score,
            "MAPE": mape_score,
            "MASE": mase_score,
            "TG": tg_score,
            "CG_EGA_AP_hypo": cg_ega_score[0],
            "CG_EGA_BE_hypo": cg_ega_score[1],
            "CG_EGA_EP_hypo": cg_ega_score[2],
            "CG_EGA_AP_eu": cg_ega_score[3],
            "CG_EGA_BE_eu": cg_ega_score[4],
            "CG_EGA_EP_eu": cg_ega_score[5],
            "CG_EGA_AP_hyper": cg_ega_score[6],
            "CG_EGA_BE_hyper": cg_ega_score[7],
            "CG_EGA_EP_hyper": cg_ega_score[8],
        }

        return results

    def plot(self, day_number=0):
        """
        Plot a given day
        :param day_number: day (int) to plot
        :return: /
        """
        cg_ega.CG_EGA(self.results[day_number], self.freq).plot(day_number)
