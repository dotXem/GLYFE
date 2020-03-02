import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from postprocessing.metrics.tools.misc import reshape_results, extract_columns_from_results
from .tools.filters import *
from .tools.misc import _all
from .p_ega import P_EGA
from .r_ega import R_EGA
from misc.constants import day_len


class CG_EGA():
    """
        The Continuous Glucose-Error Grid Analysis (CG-EGA) gives a measure of the clinical acceptability of the glucose predictions. It analyzes both the
        prediction accuracy (through the P-EGA) and the predicted variation accuracy (R-EGA).

        The implementation has been made following "Evaluating the accuracy of continuous glucose-monitoring sensors:
        continuous glucose-error grid analysis illustrated by TheraSense Freestyle Navigator data.", Kovatchev et al., 2004.
    """
    def __init__(self, results, freq):
        """
        Instantiate the CG-EGA object.
        :param results: dataframe with predictions and ground truths
        :param freq: prediction frequency in minutes (e.g., 5)
        """
        self.results = reshape_results(results, freq)
        self.freq = freq
        self.day_len = day_len // freq
        self.p_ega = P_EGA(results, freq).full()
        self.r_ega = R_EGA(results, freq).full()


    def full(self):
        """
            Full version of the CG-EGA, which consists of 3 tables (representing the hypoglycemia, euglycemia, and
            hyperglycemia regions) being the cartesian product between the P-EGA and the R-EGA. Every cell contains the
            number of predictions falling into it.

            :return: hypoglycemia full CG-EGA, euglycemia full CG-EGA, hyperglycemia full CG-EGA
        """
        y_true, y_pred, dy_true, dy_pred = extract_columns_from_results(self.results)
        p_ega, r_ega = self.p_ega, self.r_ega

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

    def simplified(self, count=False):
        """
            Simplifies the full CG-EGA into Accurate Prediction (AP), Benign Prediction (BE), and Erroneous Prediction (EP)
            rates for every glycemia regions.

            :param count: if False, the results, for every region, will be expressed as a ratio

            :return: AP rate in hypoglycemia, BE rate in hypoglycemia, EP rate in hypoglycemia,
                     AP rate in euglycemia, BE rate in euglycemia, EP rate in euglycemia,
                     AP rate in hyperglycemia, BE rate in hyperglycemia, EP rate in hyperglycemia
        """

        CG_EGA_hypo, CG_EGA_eu, CG_EGA_hyper = self.full()

        AP_hypo = np.sum(CG_EGA_hypo * filter_AP_hypo)
        BE_hypo = np.sum(CG_EGA_hypo * filter_BE_hypo)
        EP_hypo = np.sum(CG_EGA_hypo * filter_EP_hypo)

        AP_eu = np.sum(CG_EGA_eu * filter_AP_eu)
        BE_eu = np.sum(CG_EGA_eu * filter_BE_eu)
        EP_eu = np.sum(CG_EGA_eu * filter_EP_eu)

        AP_hyper = np.sum(CG_EGA_hyper * filter_AP_hyper)
        BE_hyper = np.sum(CG_EGA_hyper * filter_BE_hyper)
        EP_hyper = np.sum(CG_EGA_hyper * filter_EP_hyper)

        if not count:
            sum_hypo = (AP_hypo + BE_hypo + EP_hypo)
            sum_eu = (AP_eu + BE_eu + EP_eu)
            sum_hyper = (AP_hyper + BE_hyper + EP_hyper)


            [AP_hypo, BE_hypo, EP_hypo] = [AP_hypo / sum_hypo, BE_hypo / sum_hypo, EP_hypo / sum_hypo] if not sum_hypo == 0 else [np.nan, np.nan, np.nan]
            [AP_eu, BE_eu, EP_eu] = [AP_eu / sum_eu, BE_eu / sum_eu, EP_eu / sum_eu] if not sum_eu == 0 else [np.nan, np.nan, np.nan]
            [AP_hyper, BE_hyper, EP_hyper] = [AP_hyper / sum_hyper, BE_hyper / sum_hyper, EP_hyper / sum_hyper] if not sum_hyper == 0 else [np.nan, np.nan, np.nan]

        return AP_hypo, BE_hypo, EP_hypo, AP_eu, BE_eu, EP_eu, AP_hyper, BE_hyper, EP_hyper

    def reduced(self):
        """
            Reduces the simplified CG-EGA by not dividing the results into the glycemia regions
            :return: overall AP rate, overall BE rate, overall EP rate
        """

        AP_hypo, BE_hypo, EP_hypo, AP_eu, BE_eu, EP_eu, AP_hyper, BE_hyper, EP_hyper = self.simplified(count=True)
        sum = (AP_hypo + BE_hypo + EP_hypo + AP_eu + BE_eu + EP_eu + AP_hyper + BE_hyper + EP_hyper)
        return (AP_hypo + AP_eu + AP_hyper) / sum, (BE_hypo + BE_eu + BE_hyper) / sum, (
                EP_hypo + EP_eu + EP_hyper) / sum

    def per_sample(self):
        """
            Compute the per-sample simplified CG-EGA
            :return: pandas DataFrame with columns (y_true, y_pred, dy_true, dy_pred, P-EGA mark, R-EGA mark,
                                                    CG-EGA AP/BE/EP mark)
        """

        y_true, y_pred, dy_true, dy_pred = extract_columns_from_results(self.results)
        p_ega, r_ega = self.p_ega, self.r_ega

        df = pd.DataFrame(data=np.c_[y_true, dy_true, y_pred, dy_pred, p_ega, r_ega])
        df["CG_EGA"] = len(df.index) * ["?"]

        df.columns = ["y_true", "dy_true", "y_pred", "dy_pred", "P_A", "P_B", "P_C", "P_D", "P_E", "R_A", "R_B", "R_uC",
                      "R_lC", "R_uD", "R_lD", "R_uE", "R_lE", "CG_EGA"]

        p_ega_labels = ["A", "B", "C", "D", "E"]
        r_ega_labels = ["A", "B", "uC", "lC", "uD", "lD", "uE", "lE"]

        df["time"] = (np.arange(len(df.index)) + 1) * self.freq

        for i in range(len(p_ega)):
            p_ega_i = df.iloc[i, 4:9].values.reshape(1, -1)
            r_ega_i = df.iloc[i, 9:17].values.reshape(1, -1)
            y_true_i = df.iloc[i, 0]

            cg_ega_i = np.dot(np.transpose(r_ega_i), p_ega_i)

            if y_true_i <= 70:
                # hypoglycemia
                cg_ega_i = cg_ega_i[:, [0, 3, 4]]

                if np.sum(cg_ega_i * filter_AP_hypo) == 1:
                    df.loc[i, "CG_EGA"] = "AP"
                elif np.sum(cg_ega_i * filter_BE_hypo) == 1:
                    df.loc[i, "CG_EGA"] = "BE"
                else:
                    df.loc[i, "CG_EGA"] = "EP"

            elif y_true_i <= 180:
                # euglycemia
                cg_ega_i = cg_ega_i[:, [0, 1, 2]]

                if np.sum(cg_ega_i * filter_AP_eu) == 1:
                    df.loc[i, "CG_EGA"] = "AP"
                elif np.sum(cg_ega_i * filter_BE_eu) == 1:
                    df.loc[i, "CG_EGA"] = "BE"
                else:
                    df.loc[i, "CG_EGA"] = "EP"
            else:
                # hyperglycemia
                if np.sum(cg_ega_i * filter_AP_hyper) == 1:
                    df.loc[i, "CG_EGA"] = "AP"
                elif np.sum(cg_ega_i * filter_BE_hyper) == 1:
                    df.loc[i, "CG_EGA"] = "BE"
                else:
                    df.loc[i, "CG_EGA"] = "EP"

            df.loc[i, "P_EGA"] = p_ega_labels[np.argmax(p_ega_i.ravel())]
            df.loc[i, "R_EGA"] = r_ega_labels[np.argmax(r_ega_i.ravel())]

        df.index = pd.notnull(self.results).all(1).to_numpy().nonzero()[0]
        df_nan = pd.concat([
            self.results.copy().reset_index().rename(columns={"index": "datetime"}),
            df.loc[:, ["CG_EGA", "P_EGA", "R_EGA"]]
        ], axis=1)

        return df_nan

    def plot(self, day=0):
        """
        Plot the given day predictions and CG-EGA
        :param day: (int) number of the day for which to plot the predictions and CG-EGA
        :return: /
        """
        res = self.per_sample().iloc[day * self.day_len:(day + 1) * self.day_len - 1]
        pd.plotting.register_matplotlib_converters()
        ap = res[res["CG_EGA"] == "AP"]
        be = res[res["CG_EGA"] == "BE"]
        ep = res[res["CG_EGA"] == "EP"]

        f = plt.figure(figsize=(10, 10))
        ax1 = f.add_subplot(211)
        ax2 = f.add_subplot(223)
        ax3 = f.add_subplot(224)

        # prediction VS truth against time
        ax1.plot(res["datetime"], res["y_true"], "k", label="y_true")
        ax1.plot(res["datetime"], res["y_pred"], "--k", label="y_pred")
        ax1.plot(ap["datetime"], ap["y_pred"], label="AP", marker="o", mec="xkcd:green", mfc="xkcd:green", ls="")
        ax1.plot(be["datetime"], be["y_pred"], label="BE", marker="o", mec="xkcd:orange", mfc="xkcd:orange", ls="")
        ax1.plot(ep["datetime"], ep["y_pred"], label="EP", marker="o", mec="xkcd:red", mfc="xkcd:red", ls="")
        ax1.set_title("Prediction VS ground truth as a function of time")
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("Glucose value (mg/dL)")
        ax1.legend()

        # P-EGA structure
        ax2.plot([0, 400], [0, 400], "-k", linewidth=0.75)
        ax2.plot([58.33, 400], [58.33333 * 6 / 5, 400 * 6 / 5], "-k")
        ax2.plot([0, 58.33333], [70, 70], "-k")
        ax2.plot([70, 400], [56, 320], "-k")
        ax2.plot([70, 70], [0, 56], "-k")
        ax2.plot([70, 70], [84, 400], "-k")
        ax2.plot([0, 70], [180, 180], "-k")
        ax2.plot([70, 400], [70 * 22 / 17 + 89.412, 400 * 22 / 17 + 89.412], "-k")
        ax2.plot([180, 180], [0, 70], "-k")
        ax2.plot([180, 400], [70, 70], "-k")
        ax2.plot([240, 240], [70, 180], "-k")
        ax2.plot([240, 400], [180, 180], "-k")
        ax2.plot([130, 180], [130 * 7 / 5 - 182, 180 * 7 / 5 - 182], "-k")
        ax2.plot([130, 180], [130 * 7 / 5 - 202, 180 * 7 / 5 - 202], "--k")
        ax2.plot([180, 400], [50, 50], "--k")
        ax2.plot([240, 400], [160, 160], "--k")
        ax2.plot([58.33333, 400], [58.33333 * 6 / 5 + 20, 400 * 6 / 5 + 20], "--k")
        ax2.plot([0, 58.33333], [90, 90], "--k")
        ax2.plot([0, 70], [200, 200], "--k")
        ax2.plot([70, 400], [70 * 22 / 17 + 109.412, 400 * 22 / 17 + 109.412], "--k")

        ax2.text(38, 12, "A")
        ax2.text(12, 38, "A")
        ax2.text(375, 240, "B")
        ax2.text(260, 375, "B")
        ax2.text(150, 375, "C")
        ax2.text(165, 25, "C")
        ax2.text(25, 125, "D")
        ax2.text(375, 125, "D")
        ax2.text(375, 25, "E")
        ax2.text(25, 375, "E")

        ax2.set_xlim(0, 400)
        ax2.set_ylim(0, 400)

        ax2.set_xlabel("True glucose value [mg/dL]")
        ax2.set_ylabel("Predicted glucose value [mg/dL]")
        ax2.set_title("Point-Error Grid Analysis")

        # P-EGA data
        ax2.plot(ap["y_true"], ap["y_pred"], label="AP", marker="o", mec="xkcd:green", mfc="xkcd:green", ls="")
        ax2.plot(be["y_true"], be["y_pred"], label="BE", marker="o", mec="xkcd:orange", mfc="xkcd:orange", ls="")
        ax2.plot(ep["y_true"], ep["y_pred"], label="EP", marker="o", mec="xkcd:red", mfc="xkcd:red", ls="")

        ax2.legend()

        # R-EGA structure
        ax3.plot([-4, 4], [-4, 4], "-k", linewidth=0.75)
        ax3.plot([-4, -1, -1], [1, 1, 4], "-k")
        ax3.plot([-4, -3, 1, 1], [-1, -1, 3, 4], "-k")
        ax3.plot([-4, -2, 1, 2], [-2, -1, 2, 4], "-k")
        ax3.plot([-2, -1, 2, 4], [-4, -2, 1, 2], "-k")
        ax3.plot([-1, -1, 3, 4], [-4, -3, 1, 1], "-k")
        ax3.plot([1, 1, 4], [-4, -1, -1], "-k")

        ax3.text(-3.25, -3.75, "A")
        ax3.text(-3.75, -3.25, "A")
        ax3.text(-1.35, -3.5, "B")
        ax3.text(-3.5, -1.35, "B")
        ax3.text(0, -3.5, "C")
        ax3.text(0, 3.5, "C")
        ax3.text(-3.5, 0.5, "D")
        ax3.text(3.5, -0.5, "D")
        ax3.text(-3.5, 3.5, "E")
        ax3.text(3.5, -3.5, "E")

        ax3.set_xlim(-4, 4)
        ax3.set_ylim(-4, 4)
        ax3.set_xlabel("True glucose rate of change [mg/dL/min]")
        ax3.set_ylabel("Predicted glucose rate of change [mg/dL/min]")
        ax3.set_title("Rate-Error Grid Analysis")

        # R-EGA data
        ax3.plot(ap["dy_true"], ap["dy_pred"], label="AP", marker="o", mec="xkcd:green", mfc="xkcd:green", ls="")
        ax3.plot(be["dy_true"], be["dy_pred"], label="BE", marker="o", mec="xkcd:orange", mfc="xkcd:orange", ls="")
        ax3.plot(ep["dy_true"], ep["dy_pred"], label="EP", marker="o", mec="xkcd:red", mfc="xkcd:red", ls="")

        ax3.legend()

        plt.show()
