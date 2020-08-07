import numpy as np
from postprocessing.metrics.tools.misc import reshape_results, extract_columns_from_results
from .tools.misc import _all, _any


class P_EGA():
    """
        The Point-Error Grid Analysis (P-EGA) gives an estimation of the clinical acceptability of the glucose
        predictions based on their point-accuracy. It is also known as the Clarke Error Grid Analysis (Clarke EGA).
        Every prediction is given a mark from A to E depending of the ground truth.

        This implementation follows "Evaluating the accuracy of continuous glucose-monitoring sensors: continuous
        glucose-error grid analysis illustrated by TheraSense Freestyle Navigator data.", Kovatchev et al., 2004.
    """

    def __init__(self, results, freq):
        """
        Instantiate the P-EGA object.
        :param results: dataframe with predictions and ground truths
        :param freq: prediction frequency in minutes (e.g., 5)
        """

        self.freq = freq
        self.results = reshape_results(results, self.freq)

    def full(self):
        """
            Full version of the P-EGA, which consists of an array giving for every prediction (row), its mark vector
            (column). There are 5 columns representing the mark A, B, C, D, and E.

            :return: numy array of shape (number of predictions, 5)
        """
        y_true, y_pred, dy_true, dy_pred = extract_columns_from_results(self.results)

        # if the true rate are big, we accept bigger mistake (region borders are modified)
        mod = np.zeros_like(y_true)
        mod[_any([
            _all([
                np.greater(dy_true, -2),
                np.less_equal(dy_true, -1)]),
            _all([
                np.less(dy_true, 2),
                np.greater_equal(dy_true, 1),
            ])
        ])] = 10

        mod[_any([
            _all([
                np.less_equal(dy_true, -2)
            ]),
            _all([
                np.greater_equal(dy_true, 2)
            ])
        ])] = 20

        A = _any([
            _all([
                np.less_equal(y_pred, 70 + mod),
                np.less_equal(y_true, 70)
            ]),
            _all([
                np.less_equal(y_pred, y_true * 6 / 5 + mod),
                np.greater_equal(y_pred, y_true * 4 / 5 - mod)
            ])
        ])

        E = _any([
            _all([
                np.greater(y_true, 180),
                np.less(y_pred, 70 - mod)
            ]),
            _all([
                np.greater(y_pred, 180 + mod),
                np.less_equal(y_true, 70)
            ])
        ])

        D = _any([
            _all([
                np.greater(y_pred, 70 + mod),
                np.greater(y_pred, y_true * 6 / 5 + mod),
                np.less_equal(y_true, 70),
                np.less_equal(y_pred, 180 + mod)
            ]),
            _all([
                np.greater(y_true, 240),
                np.less(y_pred, 180 - mod),
                np.greater_equal(y_pred, 70 - mod)
            ])
        ])

        C = _any([
            _all([
                np.greater(y_true, 70),
                np.greater(y_pred, y_true * 22 / 17 + (180 - 70 * 22 / 17) + mod)
            ]),
            _all([
                np.less_equal(y_true, 180),
                np.less(y_pred, y_true * 7 / 5 - 182 - mod)
            ])
        ])

        # B being the weirdest zone in the P-EGA, we compute it last by saying
        # it's all the points that have not been classified yet.
        B = _all([
            np.equal(A, False),
            np.equal(C, False),
            np.equal(D, False),
            np.equal(E, False),
        ])

        return np.concatenate([A, B, C, D, E], axis=1)

    def mean(self):
        return np.mean(self.full(), axis=0)

    def a_plus_b(self):
        full = self.full()
        a_plus_b = full[:, 0] + full[:, 1]
        return np.sum(a_plus_b) / len(a_plus_b)
