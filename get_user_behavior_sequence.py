import datetime

import pandas as pd

from multi_pool import run_multi_pool


def get_user_behavior_sequence(i):
    training = pd.read_csv(
        f"./训练集/users/part_{i:02d}.csv",
        dtype={
            "user_id": str,
            "goods_id": str,
            "brand_id": str,
            "category_id": str
        }
    )
    training["timestamp"] = pd.to_datetime(training["timestamp"], format="%Y-%m-%d %H:%M:%S")
    training["dt"] = pd.to_datetime(training["dt"], format="%Y-%m-%d")

    training.sort_values(["user_id", "timestamp"], inplace=True, ignore_index=True)

    features = {
        "instance_id": [],
        "click_items": [],
        "click_like_items": [],
        "click_addcart_items": [],
        "click_order_items": [],
        "click_like_addcart_items": [],
        "click_like_order_items": [],
        "click_addcart_order_items": [],
        "click_like_addcart_order_items": [],
        "click_brands": [],
        "click_like_brands": [],
        "click_addcart_brands": [],
        "click_order_brands": [],
        "click_like_addcart_brands": [],
        "click_like_order_brands": [],
        "click_addcart_order_brands": [],
        "click_like_addcart_order_brands": [],
        "click_categories": [],
        "click_like_categories": [],
        "click_addcart_categories": [],
        "click_order_categories": [],
        "click_like_addcart_categories": [],
        "click_like_order_categories": [],
        "click_addcart_order_categories": [],
        "click_like_addcart_order_categories": [],
    }

    for idx, row in training.iterrows():
        if idx % 5000 == 0:
            print(idx)
            print(datetime.datetime.now())

        tmp = training[training["timestamp"].lt(row["timestamp"]) & training.user_id.eq(row["user_id"])]

        tmp_click = tmp.loc[tmp.is_like.eq(0) & tmp.is_addcart.eq(0) & tmp.is_order.eq(0)]
        tmp_click_like = tmp.loc[tmp.is_like.gt(0) & tmp.is_addcart.eq(0) & tmp.is_order.eq(0)]
        tmp_click_addcart = tmp.loc[tmp.is_like.eq(0) & tmp.is_addcart.gt(0) & tmp.is_order.eq(0)]
        tmp_click_order = tmp.loc[tmp.is_like.eq(0) & tmp.is_addcart.eq(0) & tmp.is_order.gt(0)]
        tmp_click_like_addcart = tmp.loc[tmp.is_like.gt(0) & tmp.is_addcart.gt(0) & tmp.is_order.eq(0)]
        tmp_click_like_order = tmp.loc[tmp.is_like.gt(0) & tmp.is_addcart.eq(0) & tmp.is_order.gt(0)]
        tmp_click_addcart_order = tmp.loc[tmp.is_like.eq(0) & tmp.is_addcart.gt(0) & tmp.is_order.gt(0)]
        tmp_click_like_addcart_order = tmp.loc[tmp.is_like.gt(0) & tmp.is_addcart.gt(0) & tmp.is_order.gt(0)]

        tmp_combo = [
            ["click_items", tmp_click, "goods_id"],
            ["click_brands", tmp_click, "brand_id"],
            ["click_categories", tmp_click, "category_id"],
            ["click_like_items", tmp_click_like, "goods_id"],
            ["click_like_brands", tmp_click_like, "brand_id"],
            ["click_like_categories", tmp_click_like, "category_id"],
            ["click_addcart_items", tmp_click_addcart, "goods_id"],
            ["click_addcart_brands", tmp_click_addcart, "brand_id"],
            ["click_addcart_categories", tmp_click_addcart, "category_id"],
            ["click_order_items", tmp_click_order, "goods_id"],
            ["click_order_brands", tmp_click_order, "brand_id"],
            ["click_order_categories", tmp_click_order, "category_id"],
            ["click_like_addcart_items", tmp_click_like_addcart, "goods_id"],
            ["click_like_addcart_brands", tmp_click_like_addcart, "brand_id"],
            ["click_like_addcart_categories", tmp_click_like_addcart, "category_id"],
            ["click_like_order_items", tmp_click_like_order, "goods_id"],
            ["click_like_order_brands", tmp_click_like_order, "brand_id"],
            ["click_like_order_categories", tmp_click_like_order, "category_id"],
            ["click_addcart_order_items", tmp_click_addcart_order, "goods_id"],
            ["click_addcart_order_brands", tmp_click_addcart_order, "brand_id"],
            ["click_addcart_order_categories", tmp_click_addcart_order, "category_id"],
            ["click_like_addcart_order_items", tmp_click_like_addcart_order, "goods_id"],
            ["click_like_addcart_order_brands", tmp_click_like_addcart_order, "brand_id"],
            ["click_like_addcart_order_categories", tmp_click_like_addcart_order, "category_id"]
        ]

        for a, b, c in tmp_combo:
            features[a].append("|".join(b.loc[:, c].to_list()))

        features["instance_id"].append(row["instance_id"])

    return pd.DataFrame(features)


if __name__ == "__main__":
    run_multi_pool(get_user_behavior_sequence, file_name="user_behavior_sequence")
