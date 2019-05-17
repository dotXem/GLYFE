import datetime
import sys
import itertools

def printd(*msg):
    """
    Append the datetime of the message to the message when printing. Works the same as the print() function.
    :param msg: arbitrary number of string message
    :return: /
    """
    print(datetime.datetime.now(), " ".join(str(v) for v in msg))
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