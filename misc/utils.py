import datetime
import sys
import itertools
import time
from pydoc import locate

def timeit(method):
    """
        Decorator function that print the time a function took to complete.
        :param method: function
        :return: time elapsed
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result

    return timed


def locate_model(model_name):
    return locate("processing.models." + model_name + "." + model_name.upper())


def locate_params(params_name):
    return locate ("processing.models.params." + params_name + ".parameters")


def locate_search(params_name):
    return locate ("processing.models.params." + params_name + ".search")


def printd(*msg):
    """
        Enhanced print function that prints the date and time of the log.
        :param msg: messages
        :return: /
    """
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(date, " ".join(str(v) for v in msg))
    sys.stdout.flush()

def dict_cartesian_product(dict):
    """
    From a dict, compute the cartesian product of its values.
    :param dict: dict;
    :return: array of values combinations
    """
    l = iter([])
    for hparams in [dict]:
        l2 = itertools.product(*hparams.values())
        l = itertools.chain(l, l2)
    l2 = [list(elem) for elem in l]
    return l2


def print_latex(mean, std, label=""):
    print(
        "\\textbf{" + label + "} & " + " & ".join(
            ["{0:.2f} \\scriptsize{{({1:.2f})}}".format(mean_, std_) for mean_, std_ in zip(mean, std)])
        + "\\\\")