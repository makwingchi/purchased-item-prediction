import datetime

import pandas as pd

from multi_pool import run_multi_pool


"""
Feature 8 (integrated in get_counts_rates_and_pct_7-8_13)
"""


FILE_NAME = "counts_and_rates_combo"


def get_combo_counts_and_rates(i):
    training = pd.read_csv("./训练集/training_data.csv")
    training["timestamp"] = pd.to_datetime(training["timestamp"], format="%Y-%m-%d %H:%M:%S")

    training["timestamp"] = training["timestamp"].astype(int) // 10 ** 9

    data = pd.read_csv(f"./训练集/users/part_{i:02d}.csv")
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y-%m-%d %H:%M:%S")

    data["timestamp"] = data["timestamp"].astype(int) // 10 ** 9

    features = {"instance_id": []}

    timestamp_arr = training["timestamp"].to_numpy()
    user_arr = training["user_id"].to_numpy()
    item_arr = training["goods_id"].to_numpy()
    brand_arr = training["brand_id"].to_numpy()
    category_arr = training["category_id"].to_numpy()

    click_arr = training["is_click"].to_numpy()
    like_arr = training["is_like"].to_numpy()
    addcart_arr = training["is_addcart"].to_numpy()
    order_arr = training["is_order"].to_numpy()

    for idx, row in data.iterrows():
        if idx % 500 == 0:
            print(f"{idx} of {i:02d}")
            print(datetime.datetime.now())

        before = (timestamp_arr < row["timestamp"])
        user = user_arr == row["user_id"]
        item = item_arr == row["goods_id"]
        brand = brand_arr == row["brand_id"]
        category = category_arr == row["category_id"]

        before_user_item_click = click_arr[before & user & item].sum()
        before_user_item_like = like_arr[before & user & item].sum()
        before_user_item_addcart = addcart_arr[before & user & item].sum()
        before_user_item_order = order_arr[before & user & item].sum()

        before_user_brand_click = click_arr[before & user & brand].sum()
        before_user_brand_like = like_arr[before & user & brand].sum()
        before_user_brand_addcart = addcart_arr[before & user & brand].sum()
        before_user_brand_order = order_arr[before & user & brand].sum()

        before_user_category_click = click_arr[before & user & category].sum()
        before_user_category_like = like_arr[before & user & category].sum()
        before_user_category_addcart = addcart_arr[before & user & category].sum()
        before_user_category_order = order_arr[before & user & category].sum()

        before_brand_category_click = click_arr[before & brand & category].sum()
        before_brand_category_like = like_arr[before & brand & category].sum()
        before_brand_category_addcart = addcart_arr[before & brand & category].sum()
        before_brand_category_order = order_arr[before & brand & category].sum()

        before_user_item_like_rate = before_user_item_like / (before_user_item_click + 3)
        before_user_item_addcart_rate = before_user_item_addcart / (before_user_item_click + 3)
        before_user_item_order_rate = before_user_item_order / (before_user_item_click + 3)

        before_user_brand_like_rate = before_user_brand_like / (before_user_brand_click + 3)
        before_user_brand_addcart_rate = before_user_brand_addcart / (before_user_brand_click + 3)
        before_user_brand_order_rate = before_user_brand_order / (before_user_brand_click + 3)

        before_user_category_like_rate = before_user_category_like / (before_user_category_click + 3)
        before_user_category_addcart_rate = before_user_category_addcart / (before_user_category_click + 3)
        before_user_category_order_rate = before_user_category_order / (before_user_category_click + 3)

        before_brand_category_like_rate = before_brand_category_like / (before_brand_category_click + 3)
        before_brand_category_addcart_rate = before_brand_category_addcart / (before_brand_category_click + 3)
        before_brand_category_order_rate = before_brand_category_order / (before_brand_category_click + 3)

        tmp_map = {
            "before_user_item_click": before_user_item_click,
            "before_user_item_like": before_user_item_like,
            "before_user_item_addcart": before_user_item_addcart,
            "before_user_item_order": before_user_item_order,
            "before_user_brand_click": before_user_brand_click,
            "before_user_brand_like": before_user_brand_like,
            "before_user_brand_addcart": before_user_brand_addcart,
            "before_user_brand_order": before_user_brand_order,
            "before_user_category_click": before_user_category_click,
            "before_user_category_like": before_user_category_like,
            "before_user_category_addcart": before_user_category_addcart,
            "before_user_category_order": before_user_category_order,
            "before_brand_category_click": before_brand_category_click,
            "before_brand_category_like": before_brand_category_like,
            "before_brand_category_addcart": before_brand_category_addcart,
            "before_brand_category_order": before_brand_category_order,
            "before_user_item_like_rate": before_user_item_like_rate,
            "before_user_item_addcart_rate": before_user_item_addcart_rate,
            "before_user_item_order_rate": before_user_item_order_rate,
            "before_user_brand_like_rate": before_user_brand_like_rate,
            "before_user_brand_addcart_rate": before_user_brand_addcart_rate,
            "before_user_brand_order_rate": before_user_brand_order_rate,
            "before_user_category_like_rate": before_user_category_like_rate,
            "before_user_category_addcart_rate": before_user_category_addcart_rate,
            "before_user_category_order_rate": before_user_category_order_rate,
            "before_brand_category_like_rate": before_brand_category_like_rate,
            "before_brand_category_addcart_rate": before_brand_category_addcart_rate,
            "before_brand_category_order_rate": before_brand_category_order_rate
        }

        for _k, _v in tmp_map.items():
            if _k not in features:
                features[_k] = []

            features[_k].append(_v)

    print(f"finish processing part_{i:02d}")
    pd.DataFrame(features).to_csv(f"./训练集/{FILE_NAME}/part_{i:02d}.csv", index=False)
    print(f"part_{i:02d} saved")

    return pd.DataFrame(features)


if __name__ == "__main__":
    # run_multi_pool(get_combo_counts_and_rates, file_name=FILE_NAME)
    get_combo_counts_and_rates(0)
