import datetime

import numpy as np
import pandas as pd

from multi_pool import run_multi_pool


FILE_NAME = "request_interval"


def sec_diff(a, b):
    if (a is np.nan) | (b is np.nan):
        return -1

    return a - b


"""
Features 5-6
"""


def get_request_interval(ii):
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

    data = pd.read_csv(f"./训练集/users/part_{ii:02d}.csv")
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y-%m-%d %H:%M:%S")

    data.sort_values(["user_id", "timestamp"], inplace=True, ignore_index=True)

    data["timestamp"] = data["timestamp"].astype(int) // 10 ** 9

    timestamp_arr = data["timestamp"].to_numpy()
    user_arr = data["user_id"].to_numpy()

    click_arr = data["is_click"].to_numpy()
    like_arr = data["is_like"].to_numpy()
    addcart_arr = data["is_addcart"].to_numpy()
    order_arr = data["is_order"].to_numpy()

    for idx, row in data.iterrows():
        if idx % 5000 == 0:
            print(f"{idx} of {ii:02d}")
            print(datetime.datetime.now())

        features["instance_id"].append(row["instance_id"])

        tmp = (user_arr == row["user_id"]) & (timestamp_arr <= row["timestamp"])

        click_diffs = []
        like_diffs = []
        addcart_diffs = []
        order_diffs = []

        click_tmp = timestamp_arr[tmp & (click_arr > 0)]

        if len(click_tmp) <= 1:
            click_diffs.append(-1)
        else:
            for i in range(len(click_tmp) - 1):
                click_diffs.append(sec_diff(click_tmp[i+1], click_tmp[i]))

        like_tmp = timestamp_arr[tmp & (like_arr > 0)]

        if len(like_tmp) <= 1:
            like_diffs.append(-1)
        else:
            for i in range(len(like_tmp) - 1):
                like_diffs.append(sec_diff(like_tmp[i+1], like_tmp[i]))

        addcart_tmp = timestamp_arr[tmp & (addcart_arr > 0)]

        if len(addcart_tmp) <= 1:
            addcart_diffs.append(-1)
        else:
            for i in range(len(addcart_tmp) - 1):
                addcart_diffs.append(sec_diff(addcart_tmp[i+1], addcart_tmp[i]))

        order_tmp = timestamp_arr[tmp & (order_arr > 0)]

        if len(order_tmp) <= 1:
            order_diffs.append(-1)
        else:
            for i in range(len(order_tmp) - 1):
                order_diffs.append(sec_diff(order_tmp[i+1], order_tmp[i]))

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
            diff_first_click = sec_diff(row['timestamp'], click_tmp[0])
        except:
            diff_first_click = -1

        try:
            diff_last_click = sec_diff(row['timestamp'], click_tmp[-1])
        except:
            diff_last_click = -1

        try:
            diff_first_like = sec_diff(row['timestamp'], like_tmp[0])
        except:
            diff_first_like = -1

        try:
            diff_last_like = sec_diff(row['timestamp'], like_tmp[-1])
        except:
            diff_last_like = -1

        try:
            diff_first_addcart = sec_diff(row['timestamp'], addcart_tmp[0])
        except:
            diff_first_addcart = -1

        try:
            diff_last_addcart = sec_diff(row['timestamp'], addcart_tmp[-1])
        except:
            diff_last_addcart = -1

        try:
            diff_first_order = sec_diff(row['timestamp'], order_tmp[0])
        except:
            diff_first_order = -1

        try:
            diff_last_order = sec_diff(row['timestamp'], order_tmp[-1])
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

    print(f"finish processing part_{ii:02d}")
    pd.DataFrame(features).to_csv(f"./训练集/{FILE_NAME}/part_{ii:02d}.csv", index=False)
    print(f"part_{ii:02d} saved")

    return pd.DataFrame(features)


if __name__ == "__main__":
    run_multi_pool(get_request_interval, file_name=FILE_NAME, processor=range(6))
    # get_request_interval(0)
