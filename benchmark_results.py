from misc.constants import path
from os.path import join
import os
import numpy as np
from evaluation.rmse import RMSE
from evaluation.cg_ega import AP_BE_EP
import pandas as pd
import re


def print_results(model, ph, pop):
    """
    Compute the results of a given model for a given population with a given prediction horizon.
    :param model: name of the model for which the results have been saved; e.g.: "base";
    :param ph: prediction horizon; either 30, 60 or 120;
    :param pop: population of interest, either "child", "adolescent" or "adult";
    :return: pandas DataFrame with the results (RMSE and CG-EGA) per subject and overall (mean and std)
    """
    dir = join(path, "results", model, "ph-" + str(ph))
    columns = ["RMSE", "AP_hypo", "BE_hypo", "EP_hypo", "AP_eu", "BE_eu", "EP_eu", "AP_hyper", "BE_hyper", "EP_hyper"]
    index = np.r_[[str(i) for i in np.arange(10) + 1], ["mean"], ["std"]]
    df = pd.DataFrame([], columns=columns, index=index)
    for file in os.listdir(dir):
        if pop in file:
            res = np.load(join(dir, file))

            # get patient number
            regex = ".*" + pop + "\#(\d+)_.*"
            id = int(re.search(regex, file).group(1))

            df.loc[str(id)] = np.c_[np.reshape(RMSE(*res), (1, -1)),
                                np.reshape(AP_BE_EP(*res, "hypo"), (1, -1)),
                                np.reshape(AP_BE_EP(*res, "eu"), (1, -1)),
                                np.reshape(AP_BE_EP(*res, "hyper"), (1, -1))]
    df.loc["mean"] = df.iloc[:-2, :].mean(axis=0)
    df.loc["std"] = df.iloc[:-2, :].std(axis=0)

    return df
