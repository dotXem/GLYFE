from evaluation.postprocessing import rescale
import numpy as np
from evaluation.cg_ega import AP_BE_EP
from evaluation.rmse import RMSE
import sys
import argparse
from os.path import join
from pydoc import locate
from formatting.raw_to_day_long import raw_to_day_long
from formatting.train_valid_test_split import train_valid_test_split
from preprocessing.standardization import standardization
import misc.constants as cs
from processing.nested_cross_validation import nested_cross_validation
from misc.utils import printd
from preprocessing.x_y_reshape import x_y_reshape
from pathlib import Path

""" This is the source code the benchmark GLYFE for glucose prediction in diabetes.
    For more infos on how to use it, go to its Github repository at: https://github.com/dotXem/GLYFE """


def main(args):
    # compute stdout redirection to log file
    if args.log:
        sys.stdout = open(join(cs.path, "logs", args.log + ".log"), "w")

    # retrieve the subject
    subject = args.subject if args.subject is not None else "adult#001"

    # retrieve model class from args
    model_name = args.model if args.model else "base"
    model_class = locate("models." + model_name + "." + model_name.upper())

    # retrieve the prediction horizon
    ph = args.ph if args.ph else 30

    # retrieve model's parameters
    params = locate("params." + args.params + ".parameters") if args.params \
        else locate("params." + model_name + ".parameters")
    search = locate("params." + args.params + ".search") if args.params \
        else locate("params." + model_name + ".search")

    """ DATA LOADING AND FORMATTING """
    data = raw_to_day_long(subject, params["hist"], ph)  # load and format data
    train, valid, test = train_valid_test_split(data, cv=cs.cv)  # compute processing, validation and testing sets

    """ DATA PREPROCESSING """
    train, valid, test, scalers = standardization(train, valid, test)  # standardize the data
    # train, valid, test = x_y_reshape(train, valid, test, params["hist"], ph)  # generic reshape of data for the models

    """ MODEL TRAINING, TUNING AND EVALUATION """
    results = nested_cross_validation(subject, model_class, ph, params, search, train, valid, test)

    # rescale the results
    results = rescale(results, scalers)

    # save results
    Path(join(cs.path, "results", model_name, "ph-" + str(ph))).mkdir(parents=True, exist_ok=True)
    np.save(join(cs.path, "results", model_name, "ph-" + str(ph),
                 model_name + "_ph-" + str(ph) + "_" + subject + "_results.npy"),
            np.array(results))


if __name__ == "__main__":
    """ The main function contains the following optional parameters:
            --log: file where the standard outputs will be redirected to (e.g., svr_adult#001); default: logs stay in stdout;
            --subject: subject for which the benchmark will be run (e.g., "adult#001"); default: adult_1;
            --model: model on which the benchmark will be run (e.g., "svr"); need to be lowercase; default: base;
            --ph: the prediction horizon of the models; default 30 minutes;
            --params: alternative parameters file (e.g., svr_2); default: %model%; """

    """ ARGUMENTS HANDLER """
    # retrieve and process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str)
    parser.add_argument("--subject", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--ph", type=int)
    parser.add_argument("--params", type=str)
    args = parser.parse_args()

    main(args)
