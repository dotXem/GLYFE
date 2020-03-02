import pandas as pd
import numpy as np
from datetime import timedelta, datetime


def resample(data, freq):
    """
    :param data: dataframe
    :param freq: sampling frequency
    :return: resampled data between the the first day at 00:00:00 and the last day at 23:60-freq:00 at freq sample frequency
    """
    start = data.datetime.iloc[0].strftime('%Y-%m-%d') + " 00:00:00"
    end = datetime.strptime(data.datetime.iloc[-1].strftime('%Y-%m-%d'), "%Y-%m-%d") + timedelta(days=1) - timedelta(
        minutes=freq)
    index = pd.period_range(start=start,
                            end=end,
                            freq=str(freq) + 'min').to_timestamp()
    data = data.resample(str(freq) + 'min', on="datetime").agg({'glucose': np.mean, 'CHO': np.sum, "insulin": np.sum})
    data = data.reindex(index=index)
    data = data.reset_index()
    data = data.rename(columns={"index": "datetime"})
    return data
