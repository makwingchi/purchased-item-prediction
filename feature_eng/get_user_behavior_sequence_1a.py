import datetime

import pandas as pd

from multi_pool import run_multi_pool

FILE_NAME = "user_behavior_sequence"

"""
Feature 1a
"""


def get_user_behavior_sequence(i):
    print(f"processing part_{i:02d}")

    training = pd.read_csv(
        f"./训练集/users/part_{i:02d}.csv",
        dtype={
            "goods_id": str,
            "brand_id": str,
            "category_id": str
        }
    )
    training["timestamp"] = pd.to_datetime(training["timestamp"], format="%Y-%m-%d %H:%M:%S")
    training.sort_values(["user_id", "timestamp"], inplace=True, ignore_index=True)

    training["timestamp"] = training["timestamp"].astype(int) // 10 ** 9

    features = {"instance_id": []}

    for idx, row in training.iterrows():
        if idx % 5000 == 0:  # ~19s
            print(f"{idx} of {i:02d}")
            print(datetime.datetime.now())

        tmp = training[training["timestamp"].lt(row["timestamp"]) & training.user_id.eq(row["user_id"])]

        tmp_click = tmp.loc[tmp.is_click.gt(0)]
        tmp_like = tmp.loc[tmp.is_like.gt(0)]
        tmp_addcart = tmp.loc[tmp.is_addcart.gt(0)]
        tmp_order = tmp.loc[tmp.is_order.gt(0)]

        tmp_combo = [
            ["click_items", tmp_click, "goods_id"],
            ["click_brands", tmp_click, "brand_id"],
            ["click_categories", tmp_click, "category_id"],
            ["like_items", tmp_like, "goods_id"],
            ["like_brands", tmp_like, "brand_id"],
            ["like_categories", tmp_like, "category_id"],
            ["addcart_items", tmp_addcart, "goods_id"],
            ["addcart_brands", tmp_addcart, "brand_id"],
            ["addcart_categories", tmp_addcart, "category_id"],
            ["order_items", tmp_order, "goods_id"],
            ["order_brands", tmp_order, "brand_id"],
            ["order_categories", tmp_order, "category_id"]
        ]

        for a, b, c in tmp_combo:
            if a not in features:
                features[a] = []

            features[a].append("|".join(b.loc[:, c].to_list()))

        features["instance_id"].append(row["instance_id"])

    print(f"finish processing part_{i:02d}")
    pd.DataFrame(features).to_csv(f"./训练集/{FILE_NAME}/part_{i:02d}.csv", index=False)
    print(f"part_{i:02d} saved")

    return pd.DataFrame(features)


if __name__ == "__main__":
    run_multi_pool(get_user_behavior_sequence, file_name=FILE_NAME)
