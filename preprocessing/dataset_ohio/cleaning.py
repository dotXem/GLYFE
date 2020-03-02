import numpy as np


def fill_nans(data, day_len, n_days_test):
    """
    Fill NaNs inside the dataframe of samples following:
    - CHO and insulin values are filled with 0
    - glucose history are interpolated (linearly) when possible and extrapolated if not
    :param data: sample dataframe
    :param day_len: length of day, scaled to sampling frequency
    :param n_days_test: number of test days
    :return: cleaned sample dataframe
    """
    data_nan = data.copy()

    # fill insulin and CHO nans with 0
    for col in data.columns:
        if "insulin" in col or "CHO" in col:
            data[col] = data[col].fillna(0)

    # fill glucose nans
    g_col = [col for col in data.columns if "glucose" in col]
    g_mean = data.y.mean()
    for i in range(len(data.index)):
        g_i = data.loc[i, g_col]
        if g_i.notna().all():  # no nan
            pass
        elif g_i.isna().all():  # all nan
            if i > len(data) - n_days_test * day_len + 1:
                last_known_gs = data_nan.loc[:i - 1, "glucose_0"][data_nan.loc[:i - 1, "glucose_0"].notna()]
                if len(last_known_gs) >= 1:
                    for col in g_col:
                        # data.loc[i,col] = last_known_gs.iloc[-1]
                        data.loc[i, col] = g_mean
                else:
                    for col in g_col:
                        data.loc[i, col] = g_mean
        else:  # some nan
            # compute insample nan indices, and last outsample + insample nonnan indices
            isna_idx = i + np.where(g_i.isna())[0]
            notna_idx = i + np.where(g_i.notna())[0]
            if data_nan.loc[:i - 1, "glucose_0"].notna().any():
                mask = data_nan.loc[:i - 1, "glucose_0"].notna().values
                notna_idx = np.r_[data_nan.loc[:i - 1, "glucose_0"][mask].index[-2:], notna_idx]

            # for all nan
            for isna_i in isna_idx:
                # get the two closest non nan values
                idx_diff = notna_idx - isna_i

                if np.any(idx_diff > 0) and np.any(idx_diff < 0):
                    # we got a start and an end
                    start = notna_idx[np.where(idx_diff < 0, idx_diff, -np.inf).argmax()]
                    end = notna_idx[np.where(idx_diff > 0, idx_diff, np.inf).argmin()]
                    start_val, end_val = data_nan.loc[start, "glucose_0"], data_nan.loc[end, "glucose_0"]

                    # interpolate between them
                    rate = (end_val - start_val) / (end - start)
                    data.loc[i, g_col[isna_i - i]] = data_nan.loc[start, "glucose_0"] + rate * (isna_i - start)
                elif np.any(idx_diff > 0):
                    # we only have end(s)
                    # backward extrapolation - only used in very first day where there is no start
                    if len(idx_diff) >= 2:
                        # we have two last values so we can compute a rate
                        end1, end2 = notna_idx[0], notna_idx[1]
                        end1_val, end2_val = data_nan.loc[end1, "glucose_0"], data_nan.loc[end2, "glucose_0"]
                        rate = (end2_val - end1_val) / (end2 - end1)
                        data.loc[i, g_col[isna_i - i]] = data_nan.loc[end1, "glucose_0"] - rate * (end1 - isna_i)
                    else:
                        # we have only one value so we cannot compute a rate
                        end = notna_idx[0]
                        end_val = data_nan.loc[end, "glucose_0"]
                        data.loc[i, g_col[isna_i - i]] = end_val
                elif np.any(idx_diff < 0):
                    last_val = g_i[g_i.notna()][-1]
                    data.loc[i, g_col[isna_i - i]] = last_val

    return data


def remove_nans(data):
    """
    Remove samples that still have NaNs (in y column mostly)
    :param data: dataframe of samples
    :return: no-NaN dataframe
    """
    new_data = []
    for df in data:
        new_data.append(df.dropna())
    return new_data
