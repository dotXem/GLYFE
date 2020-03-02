import misc.constants as cs
from .samples_creation import create_samples
from .splitting import split
from .standardization import standardize
from .dataset_t1dms.loading import load_t1dms
from .resampling import resample

def preprocessing_t1dms(dataset, subject, ph, hist, day_len):
    """
    T1DMS dataset preprocessing pipeline (valid for adult, adolescents and children):
    loading -> samples creation -> splitting -> standardization

    :param dataset: name of the dataset, e.g. "t1dms_adult"
    :param subject: id of the subject, e.g. "1"
    :param ph: prediction horizon, e.g. 30
    :param hist: history length, e.g. 60
    :param day_len: length of a day normalized by sampling frequency, e.g. 1440 (1440/1)
    :return: training folds, validation folds, testing folds, list of scaler (one per fold)
    """
    data = load_t1dms(dataset, subject, day_len)
    data = resample(data, cs.freq)
    data = create_samples(data, ph, hist, day_len)
    train, valid, test = split(data, day_len, cs.n_days_test, cs.cv)
    train, valid, test, scalers = standardize(train, valid, test)
    return train, valid, test, scalers