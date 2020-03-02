from .dataset_ohio.loading import load_ohio
from .dataset_ohio.cleaning import remove_nans, fill_nans
import misc.constants as cs
from .samples_creation import create_samples
from .splitting import split
from .standardization import standardize
from .resampling import resample

def preprocessing_ohio(dataset, subject, ph, hist, day_len):
    """
    OhioT1DM dataset preprocessing pipeline:
    loading -> samples creation -> cleaning (1st) -> splitting -> cleaning (2nd) -> standardization

    First cleaning is done before splitting to speedup the preprocessing

    :param dataset: name of the dataset, e.g. "ohio"
    :param subject: id of the subject, e.g. "559"
    :param ph: prediction horizon, e.g. 30
    :param hist: history length, e.g. 60
    :param day_len: length of a day normalized by sampling frequency, e.g. 288 (1440/5)
    :return: training folds, validation folds, testing folds, list of scaler (one per fold)
    """
    data = load_ohio(dataset, subject)
    data = resample(data,cs.freq)
    data = create_samples(data, ph, hist, day_len)
    data = fill_nans(data, day_len, cs.n_days_test)
    train, valid, test = split(data, day_len, cs.n_days_test, cs.cv)
    [train, valid, test] = [remove_nans(set) for set in [train, valid, test]]
    train, valid, test, scalers = standardize(train, valid, test)
    print(test[0].shape)
    return train, valid, test, scalers



