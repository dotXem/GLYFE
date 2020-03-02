from preprocessing.preprocessing_t1dms import preprocessing_t1dms
from preprocessing.preprocessing_ohio import preprocessing_ohio

preprocessing_per_dataset = {
    "t1dms_adult": preprocessing_t1dms,
    "t1dms_adolescent": preprocessing_t1dms,
    "t1dms_child": preprocessing_t1dms,
    "ohio": preprocessing_ohio,
}

def preprocessing(dataset, subject, ph, hist, day_len):
    """
    associate every dataset with a specific pipeline - which should be consistent with the others

    :param dataset: name of dataset (e.g., "ohio")
    :param subject: name of subject (e.g., "559")
    :param ph: prediction horizon in minutes (e.g., 5)
    :param hist: length of history in minutes (e.g., 60)
    :param day_len: typical length of a day in minutes standardized to the sampling frequency (e.g. 288 for 1440 min at freq=5 minutes)
    :return: train, valid, test folds
    """
    return preprocessing_per_dataset[dataset](dataset, subject, ph, hist, day_len)