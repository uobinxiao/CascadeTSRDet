import pandas as pd
import logging

def gen_blank_res_df():
    return pd.DataFrame(columns=["MAP", "AP50", "AP60", "AP70", "AP80", "AP90", "AP95", "MAPl",
                                 "MAR", "AR50", "AR60", "AR70", "AR80", "AR90", "AR95", "MARl"])


def gen_blank_res_row_col_df():
    return pd.DataFrame(columns=["MAP", "AP50", "AP60", "AP70", "AP80", "AP90", "AP95", "MAPl",
                                 "MAR", "AR50", "AR60", "AR70", "AR80", "AR90", "AR95", "MARl", "Type"])


def gen_blank_res_ic19_df():
    return pd.DataFrame(columns=["AP60", "AP70", "AP80", "AP90", "WAvg"])

def gen_blank_res_icttd_df():
    return pd.DataFrame(columns=["AP80", "AP85", "AP90", "AP95", "WAvg"])

def gen_blank_res_df_tncr():
    return pd.DataFrame(columns=["MAP", "AP50", "AP55", "AP60", "AP65", "AP70", "AP75", "AP80", "AP85", "AP90", "AP95", "MAPl",
                                 "MAR", "AR50", "AR55", "AR60", "AR65", "AR70", "AR75", "AR80", "AR85", "AR90", "AR95", "MARl"])

def gen_blank_res_df_gtc():
    return pd.DataFrame(columns=["MAP", "AP50", "AP55", "AP60", "AP65", "AP70", "AP75", "AP80", "AP85", "AP90", "AP95", "AP100",
                                 "MAR", "AR50", "AR55", "AR60", "AR65", "AR70", "AR75", "AR80", "AR85", "AR90", "AR95", "AR100"])

def print_res_df(results_df):

    logger = logging.getLogger(__name__)

    print("Epoch  MAP   AP50  AP60  AP70  AP80  AP90  AP95   MAR   AR50  AR60  AR70  AR80  AR90  AR95")

    for idx, row in results_df.iterrows():
        # if (idx % 2) == 0:
        format_str = "{:2d}  {:1.3f} {:1.3f} {:1.3f} {:1.3f} {:1.3f} {:1.3f} {:1.3f} |" +\
                        " {:1.3f} {:1.3f} {:1.3f} {:1.3f} {:1.3f} {:1.3f} {:1.3f}"

        print(format_str.format(int(idx + 1),
                                      row["MAP"], row["AP50"],row["AP60"],row["AP70"],row["AP80"],row["AP90"],row["AP95"],
                                      row["MAR"], row["AR50"],row["AR60"],row["AR70"],row["AR80"],row["AR90"],row["AR95"]))

def print_gtc_res_df(results_df):

    logger = logging.getLogger(__name__)

    print("Epoch  MAP  AP50  AP55  AP60  AP65  AP70  AP75  AP80  AP85  AP90  AP95 AP100  MAR   AR50  AR55  AR60  AR65  AR70  AR75  AR80  AR85 AR90  AR95 AR100")

    for idx, row in results_df.iterrows():
        # if (idx % 2) == 0:
        format_str = "{:2d}  {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f}|" +\
                " {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f}"

        print(format_str.format(int(idx + 1), row["MAP"], row["AP50"], row["AP55"], row["AP60"], row["AP65"], row["AP70"], row["AP75"], row["AP80"], row["AP85"], row["AP90"],row["AP95"], row["AP100"], row["MAR"], row["AR50"], row["AR55"], row["AR60"], row["AR65"], row["AR70"], row["AR75"], row["AR80"], row["AR85"], row["AR90"],row["AR95"], row["AR100"]))

def print_tncr_res_df(results_df):

    logger = logging.getLogger(__name__)

    print("Epoch  MAP  AP50  AP55  AP60  AP65  AP70  AP75  AP80  AP85  AP90  AP95   MAR   AR50  AR55  AR60  AR65  AR70  AR75  AR80  AR85 AR90  AR95")

    for idx, row in results_df.iterrows():
        # if (idx % 2) == 0:
        format_str = "{:2d}  {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f}|" +\
                " {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f} {:1.4f}"

        print(format_str.format(int(idx + 1), row["MAP"], row["AP50"], row["AP55"], row["AP60"], row["AP65"], row["AP70"], row["AP75"], row["AP80"], row["AP85"], row["AP90"],row["AP95"], row["MAR"], row["AR50"], row["AR55"], row["AR60"], row["AR65"], row["AR70"], row["AR75"], row["AR80"], row["AR85"], row["AR90"],row["AR95"]))


def print_update_ic19_res_df(results_df, combined):

    print("Epoch  F1-60  F1-70  F1-80  F1-90  |             W.Avg   Type")

    format_str1 = "{:2d}  {:1.3f} {:1.3f} {:1.3f} {:1.3f} |" + \
                  " {:1.3f} {:1.3f} {:1.3f} {:1.3f}"

    #format_str2 = "{:2d}  {:1.3f} {:1.3f} {:1.3f} {:1.3f} |" + \
    #              "                {:1.3f}     {}"

    format_str2 = "{:2d}  {:1.3f} {:1.3f} {:1.3f} {:1.3f} | {:1.3f}" 

    total_iou = 300
    counter = 1

    ic19_results_df = gen_blank_res_ic19_df()

    for idx, row in results_df.iterrows():
        # if (idx % 2) == 0:
        # format_str1 = "{:2d}  {:1.3f} {:1.3f} {:1.3f} {:1.3f} |" +\
        #                 " {:1.3f} {:1.3f} {:1.3f} {:1.3f}"

        # print(format_str1.format(int(idx + 1),
        #                               row["AP60"],row["AP70"],row["AP80"],row["AP90"],
        #                               row["AR60"],row["AR70"],row["AR80"],row["AR90"]))

        f160, f170, f180, f190 = 0, 0, 0, 0
        if row["AP60"] > 0 or row["AR60"] > 0:
            f160 = ((2 * (row["AP60"] * row["AR60"]) ) / (row["AP60"] + row["AR60"])) * 100

        if row["AP70"] > 0 or row["AR70"] > 0:
            f170 = ((2 * (row["AP70"] * row["AR70"])) / (row["AP70"] + row["AR70"])) * 100

        if row["AP80"] > 0 or row["AR80"] > 0:
            f180 = ((2 * (row["AP80"] * row["AR80"])) / (row["AP80"] + row["AR80"])) * 100

        if row["AP90"] > 0 or row["AR90"] > 0:
            f190 = ((2 * (row["AP90"] * row["AR90"])) / (row["AP90"] + row["AR90"])) * 100


        wavg = ((f160 * 60) + (f170 * 70) + (f180 * 80) + (f190 * 90)) / total_iou

        #data_type = row["Type"]
        #print(format_str2.format(counter, f160, f170, f180, f190, wavg, data_type))
        #ic19_results_df[len(ic19_results_df)] = [counter, f160, f170, f180, f190, wavg, data_type]

        print(format_str2.format(counter, f160, f170, f180, f190, wavg))
        ic19_results_df[len(ic19_results_df)] = [counter, f160, f170, f180, f190, wavg]

        if combined == True and ((idx + 1) % 2 == 0):
            counter += 1

        if combined == False:
            counter += 1


    return ic19_results_df

def print_icttd_res_df(results_df, combined):

    print("Epoch  F1-80  F1-85  F1-90  F1-95  |             W.Avg   Type")

    format_str1 = "{:2d}  {:1.3f} {:1.3f} {:1.3f} {:1.3f} |" + \
                  " {:1.3f} {:1.3f} {:1.3f} {:1.3f}"

    format_str2 = "{:2d}  {:1.3f} {:1.3f} {:1.3f} {:1.3f} | {:1.3f}" 

    counter = 1

    ic19_results_df = gen_blank_res_icttd_df()

    for idx, row in results_df.iterrows():
        f160, f170, f180, f190 = 0, 0, 0, 0
        if row["AP80"] > 0 or row["AR80"] > 0:
            f180 = ((2 * (row["AP80"] * row["AR80"]) ) / (row["AP80"] + row["AR80"])) * 100

        if row["AP85"] > 0 or row["AR85"] > 0:
            f185 = ((2 * (row["AP85"] * row["AR85"])) / (row["AP85"] + row["AR85"])) * 100

        if row["AP90"] > 0 or row["AR90"] > 0:
            f190 = ((2 * (row["AP90"] * row["AR90"])) / (row["AP90"] + row["AR90"])) * 100

        if row["AP95"] > 0 or row["AR95"] > 0:
            f195 = ((2 * (row["AP95"] * row["AR95"])) / (row["AP95"] + row["AR95"])) * 100


        wavg = ((f180 * 80) + (f185 * 85) + (f190 * 90) + (f195 * 95)) / (80.0 + 85.0 + 90.0 + 95.0)

        #data_type = row["Type"]
        #print(format_str2.format(counter, f160, f170, f180, f190, wavg, data_type))
        #ic19_results_df[len(ic19_results_df)] = [counter, f160, f170, f180, f190, wavg, data_type]

        print(format_str2.format(counter, f180, f185, f190, f195, wavg))
        ic19_results_df[len(ic19_results_df)] = [counter, f180, f185, f190, f195, wavg]

        if combined == True and ((idx + 1) % 2 == 0):
            counter += 1

        if combined == False:
            counter += 1


    return ic19_results_df
