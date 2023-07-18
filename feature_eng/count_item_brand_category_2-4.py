import datetime

import numpy as np
import pandas as pd

from multi_pool import run_multi_pool

"""
Features 2-4
"""

FILE_NAME = "item_brand_category_count"


def count_item_brand_category(i):
    data = pd.read_csv(f"./训练集/users/part_{i:02d}.csv")
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y-%m-%d %H:%M:%S")

    data.sort_values(["user_id", "timestamp"], inplace=True, ignore_index=True)
    data["timestamp"] = data["timestamp"].astype(int) // 10 ** 9

    features = {"instance_id": []}

    timestamp_arr = data["timestamp"].to_numpy()
    user_arr = data["user_id"].to_numpy()
    item_arr = data["goods_id"].to_numpy()
    brand_arr = data["brand_id"].to_numpy()
    category_arr = data["category_id"].to_numpy()

    click_arr = data["is_click"].to_numpy()
    like_arr = data["is_like"].to_numpy()
    addcart_arr = data["is_addcart"].to_numpy()
    order_arr = data["is_order"].to_numpy()

    for idx, row in data.iterrows():
        if idx % 5000 == 0:
            print(f"{idx} of {i:02d}")
            print(datetime.datetime.now())

        tmp = (user_arr == row["user_id"]) & (timestamp_arr < row["timestamp"])

        same_item = item_arr == row["goods_id"]
        same_item_click = click_arr[tmp & same_item].sum()
        same_item_like = like_arr[tmp & same_item].sum()
        same_item_addcart = addcart_arr[tmp & same_item].sum()
        same_item_order = order_arr[tmp & same_item].sum()

        same_brand = brand_arr == row["brand_id"]
        same_brand_click = click_arr[tmp & same_brand].sum()
        same_brand_like = like_arr[tmp & same_brand].sum()
        same_brand_addcart = addcart_arr[tmp & same_brand].sum()
        same_brand_order = order_arr[tmp & same_brand].sum()

        same_category = category_arr == row["category_id"]
        same_category_click = click_arr[tmp & same_category].sum()
        same_category_like = like_arr[tmp & same_category].sum()
        same_category_addcart = addcart_arr[tmp & same_category].sum()
        same_category_order = order_arr[tmp & same_category].sum()

        diff_item = item_arr != row["goods_id"]
        diff_item_click = click_arr[tmp & diff_item].sum()
        diff_item_like = like_arr[tmp & diff_item].sum()
        diff_item_addcart = addcart_arr[tmp & diff_item].sum()
        diff_item_order = order_arr[tmp & diff_item].sum()

        diff_brand = brand_arr != row["brand_id"]
        diff_brand_click = click_arr[tmp & diff_brand].sum()
        diff_brand_like = like_arr[tmp & diff_brand].sum()
        diff_brand_addcart = addcart_arr[tmp & diff_brand].sum()
        diff_brand_order = order_arr[tmp & diff_brand].sum()

        diff_category = category_arr != row["category_id"]
        diff_category_click = click_arr[tmp & diff_category].sum()
        diff_category_like = like_arr[tmp & diff_category].sum()
        diff_category_addcart = addcart_arr[tmp & diff_category].sum()
        diff_category_order = order_arr[tmp & diff_category].sum()

        item_min_time = timestamp_arr[(user_arr == row["user_id"]) & same_item].min()
        brand_min_time = timestamp_arr[(user_arr == row["user_id"]) & same_brand].min()
        category_min_time = timestamp_arr[(user_arr == row["user_id"]) & same_category].min()

        before_min_item = (timestamp_arr < item_min_time) & (user_arr == row["user_id"])
        before_min_item_click = item_arr[(click_arr > 0) & before_min_item].size
        before_min_item_like = item_arr[(like_arr > 0) & before_min_item].size
        before_min_item_addcart = item_arr[(addcart_arr > 0) & before_min_item].size
        before_min_item_order = item_arr[(order_arr > 0) & before_min_item].size

        before_min_brand = (timestamp_arr < brand_min_time) & (user_arr == row["user_id"])
        before_min_brand_click = brand_arr[(click_arr > 0) & before_min_brand].size
        before_min_brand_like = brand_arr[(like_arr > 0) & before_min_brand].size
        before_min_brand_addcart = brand_arr[(addcart_arr > 0) & before_min_brand].size
        before_min_brand_order = brand_arr[(order_arr > 0) & before_min_brand].size

        before_min_category = (timestamp_arr < category_min_time) & (user_arr == row["user_id"])
        before_min_category_click = category_arr[(click_arr > 0) & before_min_category].size
        before_min_category_like = category_arr[(like_arr > 0) & before_min_category].size
        before_min_category_addcart = category_arr[(addcart_arr > 0) & before_min_category].size
        before_min_category_order = category_arr[(order_arr > 0) & before_min_category].size

        features["instance_id"].append(row["instance_id"])

        tmp_map = {
            "same_item_click": same_item_click,
            "same_item_like": same_item_like,
            "same_item_addcart": same_item_addcart,
            "same_item_order": same_item_order,
            "same_brand_click": same_brand_click,
            "same_brand_like": same_brand_like,
            "same_brand_addcart": same_brand_addcart,
            "same_brand_order": same_brand_order,
            "same_category_click": same_category_click,
            "same_category_like": same_category_like,
            "same_category_addcart": same_category_addcart,
            "same_category_order": same_category_order,
            "diff_item_click": diff_item_click,
            "diff_item_like": diff_item_like,
            "diff_item_addcart": diff_item_addcart,
            "diff_item_order": diff_item_order,
            "diff_brand_click": diff_brand_click,
            "diff_brand_like": diff_brand_like,
            "diff_brand_addcart": diff_brand_addcart,
            "diff_brand_order": diff_brand_order,
            "diff_category_click": diff_category_click,
            "diff_category_like": diff_category_like,
            "diff_category_addcart": diff_category_addcart,
            "diff_category_order": diff_category_order,
            "before_min_item_click": before_min_item_click,
            "before_min_item_like": before_min_item_like,
            "before_min_item_addcart": before_min_item_addcart,
            "before_min_item_order": before_min_item_order,
            "before_min_brand_click": before_min_brand_click,
            "before_min_brand_like": before_min_brand_like,
            "before_min_brand_addcart": before_min_brand_addcart,
            "before_min_brand_order": before_min_brand_order,
            "before_min_category_click": before_min_category_click,
            "before_min_category_like": before_min_category_like,
            "before_min_category_addcart": before_min_category_addcart,
            "before_min_category_order": before_min_category_order
        }

        for k, v in tmp_map.items():
            if k not in features:
                features[k] = []

            features[k].append(v)

    print(f"finish processing part_{i:02d}")
    pd.DataFrame(features).to_csv(f"./训练集/{FILE_NAME}/part_{i:02d}.csv", index=False)
    print(f"part_{i:02d} saved")

    return pd.DataFrame(features)


if __name__ == "__main__":
    run_multi_pool(count_item_brand_category, file_name=FILE_NAME, processor=range(18,22))
    # count_item_brand_category(0)
