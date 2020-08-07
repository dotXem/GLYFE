import numpy as np
from datetime import datetime

def remove_last_day(data):
    """
    Simply remove the last day of data. Can be used if last day is not full.
    :param data: samples Dataframe
    :return: amputed samples Dataframe
    """
    last_day = datetime.strptime(data.datetime.iloc[-1].strftime('%Y-%m-%d'), "%Y-%m-%d").date()
    last_day_index = np.where(data.datetime.apply(lambda x: x.date()) == last_day)[0]
    data = data.drop(last_day_index,axis=0)
    return data