import misc.constants as cs
import sys
from os.path import join
from main import main
import misc.datasets
import argparse

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def batch_main():
    """
        Run main() on all the subjects of the specified dataset

        Parameters
            --dataset: which dataset to use, should be referenced in misc/datasets.py;
            --model: model on which the benchmark will be run (e.g., "svr"); need to be lowercase;
            --params: parameters of the model, usually has the same name as the model (e.g., "svr"); need to be lowercase;
            --ph: the prediction horizon of the models; default 30 minutes;
            --exp: experimental folder in which the data will be stored, inside the results directory;
            --mode: specify is the model is tested on the validation "valid" set or testing "test" set ;
            --plot: if a plot of the predictions shall be made at the end of the training;
            --log: file where the standard outputs will be redirected to; default: logs stay in stdout;

        Example:
            python main.py --dataset=ohio --model=base --params=base --ph=30
                        --exp=myexp --mode=valid --plot=1 --log=mylog
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str)
    parser.add_argument("--exp", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--ph", type=int)
    parser.add_argument("--params", type=str)
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()

    subjects = misc.datasets.datasets[args.dataset]["subjects"]

    # compute stdout redirection to log file
    if args.log:
        sys.stdout = open(join(cs.path, "logs", args.log + ".log"), "w")

    # for every combination run the main module
    for subject in subjects:
        # args = Namespace()
        main(log=args.log,
             subject=subject,
             model=args.model,
             ph=args.ph,
             params=args.params,
             exp=args.exp,
             dataset=args.dataset,
             mode=args.mode,
             plot=0)


if __name__ == "__main__":
    batch_main()
