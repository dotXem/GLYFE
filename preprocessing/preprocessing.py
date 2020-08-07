from .cleaning.unit_scaling import scaling_T1DMS
from misc import constants as cs
from preprocessing.cleaning.nans import remove_nans, fill_nans
from preprocessing.loading.loading_ohio import load_ohio
from preprocessing.loading.loading_t1dms import load_t1dms
import misc.datasets
from preprocessing.resampling import resample
from preprocessing.samples_creation import create_samples
from preprocessing.splitting import split
from preprocessing.standardization import standardize
from .cleaning.last_day_removal import remove_last_day
from .cleaning.anomalies_removal import remove_anomalies


def preprocessing_ohio(dataset, subject, ph, hist, day_len, n_days_test):
    """
    OhioT1DM dataset preprocessing pipeline:
    loading -> samples creation -> cleaning (1st) -> splitting -> cleaning (2nd) -> standardization

    First cleaning is done before splitting to speedup the preprocessing

    :param dataset: name of the dataset, e.g. "ohio"
    :param subject: id of the subject, e.g. "559"
    :param ph: prediction horizon, e.g. 30
    :param hist: history length, e.g. 60
    :param day_len: length of a day normalized by sampling frequency, e.g. 288 (1440/5)
    :return: training_old folds, validation folds, testing folds, list of scaler (one per fold)
    """
    data = load_ohio(dataset, subject)
    data = resample(data, cs.freq)
    data = create_samples(data, ph, hist, day_len)
    data = fill_nans(data, day_len, n_days_test)
    train, valid, test = split(data, day_len, n_days_test, cs.cv)
    [train, valid, test] = [remove_nans(set) for set in [train, valid, test]]
    train, valid, test, scalers = standardize(train, valid, test)
    return train, valid, test, scalers


def preprocessing_t1dms(dataset, subject, ph, hist, day_len, n_days_test):
    """
    T1DMS dataset preprocessing pipeline (valid for adult, adolescents and children):
    loading -> samples creation -> splitting -> standardization

    :param dataset: name of the dataset, e.g. "t1dms"
    :param subject: id of the subject, e.g. "1"
    :param ph: prediction horizon, e.g. 30
    :param hist: history length, e.g. 60
    :param day_len: length of a day normalized by sampling frequency, e.g. 1440 (1440/1)
    :return: training_old folds, validation folds, testing folds, list of scaler (one per fold)
    """
    data = load_t1dms(dataset, subject, day_len)
    data = scaling_T1DMS(data)
    data = resample(data, cs.freq)
    data = create_samples(data, ph, hist, day_len)
    train, valid, test = split(data, day_len, n_days_test, cs.cv)
    train, valid, test, scalers = standardize(train, valid, test)
    return train, valid, test, scalers

preprocessing_per_dataset = {
    "t1dms": preprocessing_t1dms,
    "t1dms_adult": preprocessing_t1dms,
    "t1dms_adolescent": preprocessing_t1dms,
    "t1dms_child": preprocessing_t1dms,
    "ohio": preprocessing_ohio,
}


def preprocessing(target_dataset, target_subject, ph, hist, day_len):
    """
    associate every dataset with a specific pipeline - which should be consistent with the others

    :param dataset: name of dataset (e.g., "ohio")
    :param subject: name of subject (e.g., "559")
    :param ph: prediction horizon in minutes (e.g., 5)
    :param hist: length of history in minutes (e.g., 60)
    :param day_len: typical length of a day in minutes standardized to the sampling frequency (e.g. 288 for 1440 min at freq=5 minutes)
    :return: train, valid, test folds
    """
    n_days_test = misc.datasets.datasets[target_dataset]["n_days_test"]
    return preprocessing_per_dataset[target_dataset](target_dataset, target_subject, ph, hist, day_len, n_days_test)

