from main import main
import numpy as np
from misc.utils import dict_cartesian_product

children = np.append(["child#00" + str(i) for i in range(1, 10)], "child#010")
adolescents = np.append(["adolescent#00" + str(i) for i in range(1, 10)], "adolescent#010")
adults = np.append(["adult#00" + str(i) for i in range(1, 10)], "adult#010")
all = np.r_[children, adolescents, adults]

batch_grid = {
    "log": [None],
    "subject": all,
    "model": ["base"],
    "ph": [30, 60, 120],
    "params": [None],
}


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def run_batch():
    """
    Run several main modules using the batch_grid dictionary.
    :return: /
    """

    # compute the cartesian product of the batch_params directory
    combinations = dict_cartesian_product(batch_grid)
    params_grid = [{x: vals[i] for i, x in enumerate(batch_grid.copy())} for vals in combinations]

    # for every combination run the main module
    for params in params_grid:
        args = Namespace(log=params["log"],
                         subject=params["subject"],
                         model=params["model"],
                         ph=params["ph"],
                         params=params["params"])
        main(args)


if __name__ == "__main__":
    run_batch()
