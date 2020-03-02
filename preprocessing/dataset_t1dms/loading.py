import pandas as pd
import datetime
from os.path import join
import numpy as np
import misc.constants as cs
import misc.datasets


def load_t1dms(dataset, subject, day_len):
    """
    Load a T1DMS file into a dataframe
    :param dataset: name of dataset
    :param subject: name of subject
    :param day_len: length of day scaled to sampling frequency
    :return: dataframe
    """
    df = pd.read_csv(join(cs.path, "data", dataset, subject + ".csv"), header=None, dtype=np.float64)
    df.columns = ["datetime", "glucose", "CHO", "insulin"]
    df.datetime = (df.datetime % day_len).astype("float64")
    start_day = datetime.datetime.strptime("2020-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    day_len_ds = day_len * cs.freq / misc.datasets.datasets[dataset]["freq"]
    end_day = start_day + datetime.timedelta(days=np.float64(len(df) // day_len_ds)) - datetime.timedelta(minutes=1)
    df.datetime = pd.period_range(start_day, end_day, freq='1min').to_timestamp()
    return df
