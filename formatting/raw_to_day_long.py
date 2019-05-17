import pandas as pd
from os.path import join
import numpy as np
import misc.constants as cs


def raw_to_day_long(subject, hist, ph):
    """
    Load the subject data and split the whole 29-day-long simulation into 1-day-long subsets. Then extends the subset
    given the prediction horizon (ph) and history (hist) and extract x (time, glucose, CHO, insulin) and y (glucose at PH).
    :param subject: the name of the file;
    :param hist: history (in minutes);
    :param ph: prediction horizon (in minutes);
    :return: numpy array of extended 1-day-long subsets (dataframe);
    """

    df = pd.read_csv(join("data", subject + ".csv"), header=None)
    df.columns = ["time", "glucose", "CHO", "insulin"]
    df.time = (df.time % cs.day_len).astype("float64")  # from sample number to day time

    day_long_subsets = []
    for i in range(len(df.index) // cs.day_len - 1):
        sub_df = df.iloc[(i + 1) * cs.day_len - ph - hist:(i + 2) * cs.day_len]
        day_long_subsets.append(sub_df)

    return day_long_subsets
