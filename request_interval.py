import datetime

import numpy as np
import pandas as pd

from multi_pool import run_multi_pool


def sec_diff(a, b):
    if (a is np.nan) | (b is np.nan):
        return -1

    return (datetime.datetime.strptime(str(b), "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(str(a), "%Y-%m-%d %H:%M:%S")).seconds


def get_request_interval(i):
    features = {
        "instance_id": [],
        "max_click_diff": [],
        "min_click_diff": [],
        "avg_click_diff": [],
        "med_click_diff": [],
        "max_like_diff": [],
        "min_like_diff": [],
        "avg_like_diff": [],
        "med_like_diff": [],
        "max_addcart_diff": [],
        "min_addcart_diff": [],
        "avg_addcart_diff": [],
        "med_addcart_diff": [],
        "max_order_diff": [],
        "min_order_diff": [],
        "avg_order_diff": [],
        "med_order_diff": [],
        "diff_first_click": [],
        "diff_last_click": [],
        "diff_first_like": [],
        "diff_last_like": [],
        "diff_first_addcart": [],
        "diff_last_addcart": [],
        "diff_first_order": [],
        "diff_last_order": [],
    }

    data = pd.read_csv(f"./训练集/users/part_{i:02d}.csv")
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y-%m-%d %H:%M:%S")
    data["dt"] = pd.to_datetime(data["dt"], format="%Y-%m-%d")

    for idx, row in data.iterrows():
        features["instance_id"].append(row["instance_id"])

        tmp = data[data["user_id"].eq(row["user_id"]) & data["timestamp"].le(row["timestamp"])]
        tmp.sort_values("timestamp", replace=True, ignore_index=True)

        click_diffs = []
        like_diffs = []
        addcart_diffs = []
        order_diffs = []

        if len(tmp) <= 1:
            click_diffs.append(-1)
        else:
            for i in range(len(tmp) - 1):
                click_diffs.append(sec_diff(tmp.loc[i, 'timestamp'], tmp.loc[i+1, 'timestamp']))

        like_tmp = tmp[tmp["is_like"].gt(0)].reset_index(drop=True)

        if len(like_tmp) <= 1:
            like_diffs.append(-1)
        else:
            for i in range(len(like_tmp) - 1):
                like_diffs.append(sec_diff(like_tmp.loc[i, 'timestamp'], like_tmp.loc[i+1, 'timestamp']))

        addcart_tmp = tmp[tmp["is_addcart"].gt(0)].reset_index(drop=True)

        if len(addcart_tmp) <= 1:
            addcart_diffs.append(-1)
        else:
            for i in range(len(addcart_tmp) - 1):
                addcart_diffs.append(sec_diff(addcart_tmp.loc[i, 'timestamp'], addcart_tmp.loc[i+1, 'timestamp']))

        order_tmp = tmp[tmp["is_order"].gt(0)].reset_index(drop=True)

        if len(order_tmp) <= 1:
            order_diffs.append(-1)
        else:
            for i in range(len(order_tmp) - 1):
                order_diffs.append(sec_diff(order_tmp.loc[i, 'timestamp'], order_tmp.loc[i+1, 'timestamp']))

        features["max_click_diff"].append(np.max(click_diffs))
        features["min_click_diff"].append(np.min(click_diffs))
        features["avg_click_diff"].append(np.mean(click_diffs))
        features["med_click_diff"].append(np.median(click_diffs))

        features["max_like_diff"].append(np.max(like_diffs))
        features["min_like_diff"].append(np.min(like_diffs))
        features["avg_like_diff"].append(np.mean(like_diffs))
        features["med_like_diff"].append(np.median(like_diffs))

        features["max_addcart_diff"].append(np.max(addcart_diffs))
        features["min_addcart_diff"].append(np.min(addcart_diffs))
        features["avg_addcart_diff"].append(np.mean(addcart_diffs))
        features["med_addcart_diff"].append(np.median(addcart_diffs))

        features["max_order_diff"].append(np.max(order_diffs))
        features["min_order_diff"].append(np.min(order_diffs))
        features["avg_order_diff"].append(np.mean(order_diffs))
        features["med_order_diff"].append(np.median(order_diffs))

        try:
            diff_first_click = sec_diff(row['timestamp'], tmp.loc[0, 'timestamp'])
        except:
            diff_first_click = -1

        try:
            diff_last_click = sec_diff(row['timestamp'], tmp.loc[len(tmp) - 1, 'timestamp'])
        except:
            diff_last_click = -1

        try:
            diff_first_like = sec_diff(row['timestamp'], like_tmp.loc[0, 'timestamp'])
        except:
            diff_first_like = -1

        try:
            diff_last_like = sec_diff(row['timestamp'], like_tmp.loc[len(like_tmp) - 1, 'timestamp'])
        except:
            diff_last_like = -1

        try:
            diff_first_addcart = sec_diff(row['timestamp'], addcart_tmp.loc[0, 'timestamp'])
        except:
            diff_first_addcart = -1

        try:
            diff_last_addcart = sec_diff(row['timestamp'], addcart_tmp.loc[len(addcart_tmp) - 1, 'timestamp'])
        except:
            diff_last_addcart = -1

        try:
            diff_first_order = sec_diff(row['timestamp'], order_tmp.loc[0, 'timestamp'])
        except:
            diff_first_order = -1

        try:
            diff_last_order = sec_diff(row['timestamp'], order_tmp.loc[len(order_tmp) - 1, 'timestamp'])
        except:
            diff_last_order = -1

        features["diff_first_click"].append(diff_first_click)
        features["diff_last_click"].append(diff_last_click)
        features["diff_first_like"].append(diff_first_like)
        features["diff_last_like"].append(diff_last_like)
        features["diff_first_addcart"].append(diff_first_addcart)
        features["diff_last_addcart"].append(diff_last_addcart)
        features["diff_first_order"].append(diff_first_order)
        features["diff_last_order"].append(diff_last_order)

    return pd.DataFrame(features)


if __name__ == "__main__":
    run_multi_pool(get_request_interval, "request_interval")