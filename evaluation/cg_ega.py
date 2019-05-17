import numpy as np
from misc.constants import day_len, freq


def R_EGA(y_true, y_pred):
    """
    Compute the Rate-Error Grid Analysis from "Evaluating the accuracy of continuous glucose-monitoring sensors:
    continuous glucose-error grid analysis illustrated by TheraSense Freestyle Navigator data.", Kovatchev et al., 2004.
    :param y_true: true glucose values;
    :param y_pred: predicted glucose values;
    :param hist: amount of history use to make the predictions;
    :param ph: prediction horizon;
    :return: numpy array of R-EGA prediction classification;
    """
    y_true, y_pred = _daily_reshape(y_true, y_pred)

    dy_true, _ = _derivatives(y_true)
    dy_pred, _ = _derivatives(y_pred)

    A = _any([
        _all([  # upper and lower
            np.greater_equal(dy_pred, dy_true - 1),
            np.less_equal(dy_pred, dy_true + 1)
        ]),
        _all([  # left
            np.less_equal(dy_pred, dy_true / 2),
            np.greater_equal(dy_pred, dy_true * 2)
        ]),
        _all([  # right
            np.less_equal(dy_pred, dy_true * 2),
            np.greater_equal(dy_pred, dy_true / 2)
        ])
    ])

    B = _all([
        np.equal(A, False),  # not in A but satisfies the cond below
        _any([
            _all([
                np.less_equal(dy_pred, -1),
                np.less_equal(dy_true, -1)
            ]),
            _all([
                np.less_equal(dy_pred, dy_true + 2),
                np.greater_equal(dy_pred, dy_true - 2)
            ]),
            _all([
                np.greater_equal(dy_pred, 1),
                np.greater_equal(dy_true, 1)
            ])
        ])
    ])

    uC = _all([
        np.less(dy_true, 1),
        np.greater_equal(dy_true, -1),
        np.greater(dy_pred, dy_true + 2)
    ])

    lC = _all([
        np.less_equal(dy_true, 1),
        np.greater(dy_true, -1),
        np.less(dy_pred, dy_true - 2)
    ])

    uD = _all([
        np.less_equal(dy_pred, 1),
        np.greater_equal(dy_pred, -1),
        np.greater(dy_pred, dy_true + 2)
    ])

    lD = _all([
        np.less_equal(dy_pred, 1),
        np.greater_equal(dy_pred, -1),
        np.less(dy_pred, dy_true - 2)
    ])

    uE = _all([
        np.greater(dy_pred, 1),
        np.less(dy_true, -1)
    ])

    lE = _all([
        np.less(dy_pred, -1),
        np.greater(dy_true, 1)
    ])

    return np.concatenate([A, B, uC, lC, uD, lD, uE, lE], axis=1)


def P_EGA(y_true, y_pred):
    """
    Compute the Prediction-Error Grid Analysis from "Evaluating the accuracy of continuous glucose-monitoring sensors:
    continuous glucose-error grid analysis illustrated by TheraSense Freestyle Navigator data.", Kovatchev et al., 2004.
    :param y_true: true glucose values;
    :param y_pred: predicted glucose values;
    :return: numpy array of P-EGA prediction classification;
    """
    y_true, y_pred = _daily_reshape(y_true, y_pred)

    dy_true, y_true = _derivatives(y_true)
    _, y_pred = _derivatives(y_pred)

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


def CG_EGA(y_true, y_pred):
    """
    Compute the Continuous Glucose-Error Grid Analysis from "Evaluating the accuracy of continuous glucose-monitoring
    sensors: continuous glucose-error grid analysis illustrated by TheraSense Freestyle Navigator data.", Kovatchev et
    al., 2004. It is computed by combining the R-EGA and P-EGA.
    :param y_true: true glucose values;
    :param y_pred: predicted glucose values;
    :param hist: amount of history use to make the predictions;
    :param ph: prediction horizon;
    :return: numpy array of CG-EGA classification, divided by glycemia regions (hypo, eu, hyper);
    """
    y_true, y_pred = _daily_reshape(y_true, y_pred)

    r_ega = R_EGA(y_true, y_pred)
    p_ega = P_EGA(y_true, y_pred)

    _, y_true = _derivatives(y_true)
    # y_pred, _ = _derivatives(y_pred)

    # compute the glycemia regions
    hypoglycemia = np.less_equal(y_true, 70).reshape(-1, 1)
    euglycemia = _all([
        np.less_equal(y_true, 180),
        np.greater(y_true, 70)
    ]).reshape(-1, 1)
    hyperglycemia = np.greater(y_true, 180).reshape(-1, 1)

    # apply region filter and convert to 0s and 1s
    P_hypo = np.reshape(np.concatenate([np.reshape(p_ega[:, 0], (-1, 1)),
                                        np.reshape(p_ega[:, 3], (-1, 1)),
                                        np.reshape(p_ega[:, 4], (-1, 1))], axis=1).astype("int32") *
                        hypoglycemia.astype("int32"),
                        (-1, 3))
    P_eu = np.reshape(np.concatenate([np.reshape(p_ega[:, 0], (-1, 1)),
                                      np.reshape(p_ega[:, 1], (-1, 1)),
                                      np.reshape(p_ega[:, 2], (-1, 1))], axis=1).astype("int32") *
                      euglycemia.astype("int32"),
                      (-1, 3))
    P_hyper = np.reshape(p_ega.astype("int32") * hyperglycemia.astype("int32"), (-1, 5))

    R_hypo = np.reshape(r_ega.astype("int32") * hypoglycemia.astype("int32"), (-1, 8))
    R_eu = np.reshape(r_ega.astype("int32") * euglycemia.astype("int32"), (-1, 8))
    R_hyper = np.reshape(r_ega.astype("int32") * hyperglycemia.astype("int32"), (-1, 8))

    CG_EGA_hypo = np.dot(np.transpose(R_hypo), P_hypo)
    CG_EGA_eu = np.dot(np.transpose(R_eu), P_eu)
    CG_EGA_hyper = np.dot(np.transpose(R_hyper), P_hyper)

    return CG_EGA_hypo, CG_EGA_eu, CG_EGA_hyper


def AP_BE_EP_count(y_true, y_pred):
    """
    Simplify the CG-EGA results into Accurate Predictions, Benign Errors and Erroneous Predictions, following
    Compute the Rate-Error Grid Analysis from "Evaluating the accuracy of continuous glucose-monitoring sensors:
    continuous glucose-error grid analysis illustrated by TheraSense Freestyle Navigator data.", Kovatchev et al., 2004.
    :param y_true: true glucose values;
    :param y_pred: predicted glucose values;
    :param hist: amount of history use to make the predictions;
    :param ph: prediction horizon;
    :return: numpy array count of predictions per AB/BE/EP in every glycemia region (hypo, eu, hyper);
    """
    y_true, y_pred = _daily_reshape(y_true, y_pred)

    CG_EGA_hypo, CG_EGA_eu, CG_EGA_hyper = CG_EGA(y_true, y_pred)

    filter_AP_hypo = [
        [1, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    filter_BE_hypo = [
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
    ]

    filter_EP_hypo = [
        [0, 1, 1],
        [0, 1, 1],
        [0, 1, 1],
        [0, 1, 1],
        [1, 1, 1],
        [0, 1, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]

    AP_hypo = np.sum(CG_EGA_hypo * filter_AP_hypo)
    BE_hypo = np.sum(CG_EGA_hypo * filter_BE_hypo)
    EP_hypo = np.sum(CG_EGA_hypo * filter_EP_hypo)

    filter_AP_eu = [
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    filter_BE_eu = [
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 0],
        [1, 1, 0],
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    filter_EP_eu = [
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]

    AP_eu = np.sum(CG_EGA_eu * filter_AP_eu)
    BE_eu = np.sum(CG_EGA_eu * filter_BE_eu)
    EP_eu = np.sum(CG_EGA_eu * filter_EP_eu)

    filter_AP_hyper = [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]

    filter_BE_hyper = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]

    filter_EP_hyper = [
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]

    AP_hyper = np.sum(CG_EGA_hyper * filter_AP_hyper)
    BE_hyper = np.sum(CG_EGA_hyper * filter_BE_hyper)
    EP_hyper = np.sum(CG_EGA_hyper * filter_EP_hyper)

    return AP_hypo, BE_hypo, EP_hypo, AP_eu, BE_eu, EP_eu, AP_hyper, BE_hyper, EP_hyper


def AP_BE_EP(y_true, y_pred, region=None):
    """
    From the AP_BE_EP_count function, compute the rates of every categories;
    :param y_true: true glucose values;
    :param y_pred: predicted glucose values;
    :param hist: amount of history use to make the predictions;
    :param ph: prediction horizon;
    :param region: region on which we compute the rates; accept "hypo", "eu", "hyper", "all", None. None = all;
    :return: numpy array of R-EGA prediction classification;
    """
    AP_hypo, BE_hypo, EP_hypo, AP_eu, BE_eu, EP_eu, AP_hyper, BE_hyper, EP_hyper = AP_BE_EP_count(y_true, y_pred)
    if region == "hypo":
        sum = (AP_hypo + BE_hypo + EP_hypo)
        return AP_hypo / sum, BE_hypo / sum, EP_hypo / sum
    elif region == "eu":
        sum = (AP_eu + BE_eu + EP_eu)
        return AP_eu / sum, BE_eu / sum, EP_eu / sum
    elif region == "hyper":
        sum = (AP_hyper + BE_hyper + EP_hyper)
        return AP_hyper / sum, BE_hyper / sum, EP_hyper / sum
    elif region == "all" or region is None:
        sum = (AP_hypo + BE_hypo + EP_hypo + AP_eu + BE_eu + EP_eu + AP_hyper + BE_hyper + EP_hyper)
        return (AP_hypo + AP_eu + AP_hyper) / sum, (BE_hypo + BE_eu + BE_hyper) / sum, (
                EP_hypo + EP_eu + EP_hyper) / sum


def AB_P_EGA(y_true, y_pred):
    """
    Compute the rates of predictions falling into the A+B categories in the P-EGA.
    :param y_true: true glucose values;
    :param y_pred: predicted glucose values;
    :param hist: amount of history use to make the predictions;
    :param ph: prediction horizon;
    :return: rate of predictions falling into the A+B range in the P-EGA;
    """
    p_ega = P_EGA(y_true, y_pred)
    p_ega = p_ega.astype("int32")
    a_count = np.sum(p_ega[:, 0])
    b_count = np.sum(p_ega[:, 1])
    return (a_count + b_count) / (np.sum(p_ega))


def AB_R_EGA(y_true, y_pred):
    """
    Compute the rates of predictions falling into the A+B categories in the R-EGA.
    :param y_true: true glucose values;
    :param y_pred: predicted glucose values;
    :param hist: amount of history use to make the predictions;
    :param ph: prediction horizon;
    :return: rate of predictions falling into the A+B range in the R-EGA;
    """
    r_ega = R_EGA(y_true, y_pred)
    r_ega = r_ega.astype("int32")
    a_count = np.sum(r_ega[:, 0])
    b_count = np.sum(r_ega[:, 1])
    return (a_count + b_count) / (np.sum(r_ega))


def cg_ega_fast_plotting(y_true, y_pred):
    """
    (Fast) Plotting function of the P-EGA and R-EGA
    :param y_true: true glucose values;
    :param y_pred: predicted glucose values;
    :param hist: amount of history use to make the predictions;
    :param ph: prediction horizon;
    :return: /
    """
    y_true, y_pred = _daily_reshape(y_true, y_pred)

    r_ega = R_EGA(y_true, y_pred)
    p_ega = P_EGA(y_true, y_pred)

    dy_true, y_true = _derivatives(y_true)
    dy_pred, y_pred = _derivatives(y_pred)

    import matplotlib.pyplot as plt
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("P-EGA")
    for i, marker in enumerate(["$A$", "$B$", "$C$", "$D$", "$E$"]):
        ax1.plot(y_true[p_ega[:, i]], y_pred[p_ega[:, i]], marker=marker, linestyle="None")
    ax1.set_xlim([0, 400])
    ax1.set_ylim([0, 400])

    ax2.set_title("R-EGA")
    for i, marker in enumerate(["$A$", "$B$", "$uC$", "$lC$", "$uD$", "$lD$", "$uE$", "$lE$"]):
        ax2.plot(dy_true[r_ega[:, i]], dy_pred[r_ega[:, i]], marker=marker, linestyle="None")
    ax2.set_xlim([-4, 4])
    ax2.set_ylim([-4, 4])


def _daily_reshape(trues, preds):
    """
    Reshape the predictions and true glucose values into a day format
    :param trues: 
    :param preds: 
    :param hist: 
    :param ph: 
    :return: 
    """
    trues = trues.reshape(-1, day_len)
    preds = preds.reshape(-1, day_len)

    return trues, preds


def _derivatives(y):
    """
    Compute the derivative of the signal y, (y[i] - y[i-1]) / (t[i] - t[i-1]).
    :param y: signal, either true or predicted glucose values;
    :return: the derivatives and its associated signal (every sample has a corresponding derivative)
    """
    dy = np.reshape(np.diff(y, axis=1), (-1, 1)) / freq
    y = np.reshape(y[:, 1:], (-1, 1))
    return dy, y


def _any(l, axis=1):
    return np.reshape(np.any(np.concatenate(l, axis=axis), axis=axis), (-1, 1))


def _all(l, axis=1):
    return np.reshape(np.all(np.concatenate(l, axis=axis), axis=axis), (-1, 1))