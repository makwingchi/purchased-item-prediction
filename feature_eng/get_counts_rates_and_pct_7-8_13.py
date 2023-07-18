import datetime

import pandas as pd

from multi_pool import run_multi_pool


"""
Features 7-8, 13 (removed; use half-day split instead; check get_counts_rates_rank_and_pct_7-13.py)
"""


FILE_NAME = "counts_and_rates"


def get_counts_rates_and_pct(i):
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

        before = timestamp_arr < row["timestamp"]
        user = user_arr == row["user_id"]
        item = item_arr == row["goods_id"]
        brand = brand_arr == row["brand_id"]
        category = category_arr == row["category_id"]

        # self/feature set 7 (#click, #like, #addcart, #order, like%, addcart%, order%)
        before_user = before & user
        before_user_click = click_arr[before_user].sum()
        before_user_like = like_arr[before_user].sum()
        before_user_addcart = addcart_arr[before_user].sum()
        before_user_order = order_arr[before_user].sum()

        before_item = before & item
        before_item_click = click_arr[before_item].sum()
        before_item_like = like_arr[before_item].sum()
        before_item_addcart = addcart_arr[before_item].sum()
        before_item_order = order_arr[before_item].sum()

        before_brand = before & brand
        before_brand_click = click_arr[before_brand].sum()
        before_brand_like = like_arr[before_brand].sum()
        before_brand_addcart = addcart_arr[before_brand].sum()
        before_brand_order = order_arr[before_brand].sum()

        before_category = before & category
        before_category_click = click_arr[before_category].sum()
        before_category_like = like_arr[before_category].sum()
        before_category_addcart = addcart_arr[before_category].sum()
        before_category_order = order_arr[before_category].sum()

        before_user_like_rate = before_user_like / (before_user_click + 3)
        before_user_addcart_rate = before_user_addcart / (before_user_click + 3)
        before_user_order_rate = before_user_order / (before_user_click + 3)

        before_item_like_rate = before_item_like / (before_item_click + 3)
        before_item_addcart_rate = before_item_addcart / (before_item_click + 3)
        before_item_order_rate = before_item_order / (before_item_click + 3)

        before_brand_like_rate = before_brand_like / (before_brand_click + 3)
        before_brand_addcart_rate = before_brand_addcart / (before_brand_click + 3)
        before_brand_order_rate = before_brand_order / (before_brand_click + 3)

        before_category_like_rate = before_category_like / (before_category_click + 3)
        before_category_addcart_rate = before_category_addcart / (before_category_click + 3)
        before_category_order_rate = before_category_order / (before_category_click + 3)

        # combo/feature set 8 (#click, #like, #addcart, #order, like%, addcart%, order%)
        before_user_item = before_user & item
        before_user_item_click = click_arr[before_user_item].sum()
        before_user_item_like = like_arr[before_user_item].sum()
        before_user_item_addcart = addcart_arr[before_user_item].sum()
        before_user_item_order = order_arr[before_user_item].sum()

        before_user_brand = before_user & brand
        before_user_brand_click = click_arr[before_user_brand].sum()
        before_user_brand_like = like_arr[before_user_brand].sum()
        before_user_brand_addcart = addcart_arr[before_user_brand].sum()
        before_user_brand_order = order_arr[before_user_brand].sum()

        before_user_category = before_user & category
        before_user_category_click = click_arr[before_user_category].sum()
        before_user_category_like = like_arr[before_user_category].sum()
        before_user_category_addcart = addcart_arr[before_user_category].sum()
        before_user_category_order = order_arr[before_user_category].sum()

        before_brand_category = before_brand & category
        before_brand_category_click = click_arr[before_brand_category].sum()
        before_brand_category_like = like_arr[before_brand_category].sum()
        before_brand_category_addcart = addcart_arr[before_brand_category].sum()
        before_brand_category_order = order_arr[before_brand_category].sum()

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

        # pct/feature set 13
        user_item_user_click = before_user_item_click / (before_user_click + 3)
        user_item_user_like = before_user_item_like / (before_user_like + 3)
        user_item_user_addcart = before_user_item_addcart / (before_user_addcart + 3)
        user_item_user_order = before_user_item_order / (before_user_order + 3)

        user_item_item_click = before_user_item_click / (before_item_click + 3)
        user_item_item_like = before_user_item_like / (before_item_like + 3)
        user_item_item_addcart = before_user_item_addcart / (before_item_addcart + 3)
        user_item_item_order = before_user_item_order / (before_item_order + 3)

        user_brand_user_click = before_user_brand_click / (before_user_click + 3)
        user_brand_user_like = before_user_brand_like / (before_user_like + 3)
        user_brand_user_addcart = before_user_brand_addcart / (before_user_addcart + 3)
        user_brand_user_order = before_user_brand_order / (before_user_order + 3)

        user_brand_brand_click = before_user_brand_click / (before_brand_click + 3)
        user_brand_brand_like = before_user_brand_like / (before_brand_like + 3)
        user_brand_brand_addcart = before_user_brand_addcart / (before_brand_addcart + 3)
        user_brand_brand_order = before_user_brand_order / (before_brand_order + 3)

        user_category_user_click = before_user_category_click / (before_user_click + 3)
        user_category_user_like = before_user_category_like / (before_user_like + 3)
        user_category_user_addcart = before_user_category_addcart / (before_user_addcart + 3)
        user_category_user_order = before_user_category_order / (before_user_order + 3)

        user_category_category_click = before_user_category_click / (before_category_click + 3)
        user_category_category_like = before_user_category_like / (before_category_like + 3)
        user_category_category_addcart = before_user_category_addcart / (before_category_addcart + 3)
        user_category_category_order = before_user_category_order / (before_category_order + 3)

        brand_category_brand_click = before_brand_category_click / (before_brand_click + 3)
        brand_category_brand_like = before_brand_category_like / (before_brand_like + 3)
        brand_category_brand_addcart = before_brand_category_addcart / (before_brand_addcart + 3)
        brand_category_brand_order = before_brand_category_order / (before_brand_order + 3)

        brand_category_category_click = before_brand_category_click / (before_category_click + 3)
        brand_category_category_like = before_brand_category_like / (before_category_like + 3)
        brand_category_category_addcart = before_brand_category_addcart / (before_category_addcart + 3)
        brand_category_category_order = before_brand_category_order / (before_category_order + 3)

        features["instance_id"].append(row["instance_id"])

        tmp_dict = {
            # self
            "before_user_click": before_user_click,
            "before_user_like": before_user_like,
            "before_user_addcart": before_user_addcart,
            "before_user_order": before_user_order,
            "before_item_click": before_item_click,
            "before_item_like": before_item_like,
            "before_item_addcart": before_item_addcart,
            "before_item_order": before_item_order,
            "before_brand_click": before_brand_click,
            "before_brand_like": before_brand_like,
            "before_brand_addcart": before_brand_addcart,
            "before_brand_order": before_brand_order,
            "before_category_click": before_category_click,
            "before_category_like": before_category_like,
            "before_category_addcart": before_category_addcart,
            "before_category_order": before_category_order,
            "before_user_like_rate": before_user_like_rate,
            "before_user_addcart_rate": before_user_addcart_rate,
            "before_user_order_rate": before_user_order_rate,
            "before_item_like_rate": before_item_like_rate,
            "before_item_addcart_rate": before_item_addcart_rate,
            "before_item_order_rate": before_item_order_rate,
            "before_brand_like_rate": before_brand_like_rate,
            "before_brand_addcart_rate": before_brand_addcart_rate,
            "before_brand_order_rate": before_brand_order_rate,
            "before_category_like_rate": before_category_like_rate,
            "before_category_addcart_rate": before_category_addcart_rate,
            "before_category_order_rate": before_category_order_rate,
            # combo
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
            "before_brand_category_order_rate": before_brand_category_order_rate,
            # pct
            "user_item_user_click": user_item_user_click,
            "user_item_user_like": user_item_user_like,
            "user_item_user_addcart": user_item_user_addcart,
            "user_item_user_order": user_item_user_order,
            "user_item_item_click": user_item_item_click,
            "user_item_item_like": user_item_item_like,
            "user_item_item_addcart": user_item_item_addcart,
            "user_item_item_order": user_item_item_order,
            "user_brand_user_click": user_brand_user_click,
            "user_brand_user_like": user_brand_user_like,
            "user_brand_user_addcart": user_brand_user_addcart,
            "user_brand_user_order": user_brand_user_order,
            "user_brand_brand_click": user_brand_brand_click,
            "user_brand_brand_like": user_brand_brand_like,
            "user_brand_brand_addcart": user_brand_brand_addcart,
            "user_brand_brand_order": user_brand_brand_order,
            "user_category_user_click": user_category_user_click,
            "user_category_user_like": user_category_user_like,
            "user_category_user_addcart": user_category_user_addcart,
            "user_category_user_order": user_category_user_order,
            "user_category_category_click": user_category_category_click,
            "user_category_category_like": user_category_category_like,
            "user_category_category_addcart": user_category_category_addcart,
            "user_category_category_order": user_category_category_order,
            "brand_category_brand_click": brand_category_brand_click,
            "brand_category_brand_like": brand_category_brand_like,
            "brand_category_brand_addcart": brand_category_brand_addcart,
            "brand_category_brand_order": brand_category_brand_order,
            "brand_category_category_click": brand_category_category_click,
            "brand_category_category_like": brand_category_category_like,
            "brand_category_category_addcart": brand_category_category_addcart,
            "brand_category_category_order": brand_category_category_order
        }

        for _k, _v in tmp_dict.items():
            if _k not in features:
                features[_k] = []

            features[_k].append(_v)

    print(f"finish processing part_{i:02d}")
    pd.DataFrame(features).to_csv(f"./训练集/{FILE_NAME}/part_{i:02d}.csv", index=False)
    print(f"part_{i:02d} saved")

    return pd.DataFrame(features)


if __name__ == "__main__":
    # run_multi_pool(get_counts_rates_and_pct, file_name=FILE_NAME, processor=range(6))
    get_counts_rates_and_pct(0)
