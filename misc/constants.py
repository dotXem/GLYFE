seed = 1  # random seed of 1
cv = 5  # outer cross-validation factor 4
path = "."  # current path, change if the main function is run from outside the project
freq = 5
day_len = 1440  # number of samples per day, 1440 = 1 sample every minute
day_len_freq = day_len // freq
n_days_test = 10 # number of last days taken as test set
