from collections import Counter

import pandas as pd

from multi_pool import run_multi_pool

"""
Feature 1b  (removed)
"""


@DeprecationWarning
def get_top_user_behavior_sequence(i):
    training = pd.read_csv("../训练集/training_data.csv")
    training["timestamp"] = pd.to_datetime(training["timestamp"], format="%Y-%m-%d %H:%M:%S")
    training["dt"] = pd.to_datetime(training["dt"], format="%Y%m%d")

    data = pd.read_csv(f"./训练集/users/part_{i:02d}.csv")
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y-%m-%d %H:%M:%S")
    data["dt"] = pd.to_datetime(data["dt"], format="%Y-%m-%d")

    features = {
        "instance_id": [],
        "item_click": [],
        "item_like": [],
        "item_addcart": [],
        "item_order": [],
        "brand_click": [],
        "brand_like": [],
        "brand_addcart": [],
        "brand_order": [],
        "category_click": [],
        "category_like": [],
        "category_addcart": [],
        "category_order": [],
    }

    for index, row in data.iterrows():
        before = training[training["timestamp"].lt(row["timestamp"])].reset_index(drop=True)

        before_click = before[before["is_click"].gt(0)]
        before_like = before[before["is_like"].gt(0)]
        before_addcart = before[before["is_addcart"].gt(0)]
        before_order = before[before["is_order"].gt(0)]

        item_count = {
            "click": Counter(before_click["goods_id"].to_list()),
            "like": Counter(before_like["goods_id"].to_list()),
            "addcart": Counter(before_addcart["goods_id"].to_list()),
            "order": Counter(before_order["goods_id"].to_list())
        }

        brand_count = {
            "click": Counter(before_click["brand_id"].to_list()),
            "like": Counter(before_like["brand_id"].to_list()),
            "addcart": Counter(before_addcart["brand_id"].to_list()),
            "order": Counter(before_order["brand_id"].to_list())
        }

        category_count = {
            "click": Counter(before_click["category_id"].to_list()),
            "like": Counter(before_like["category_id"].to_list()),
            "addcart": Counter(before_addcart["category_id"].to_list()),
            "order": Counter(before_order["category_id"].to_list())
        }

        item_rank = {
            "click": {pair[0]: idx + 1 for idx, pair in enumerate(item_count["click"].most_common())},
            "like": {pair[0]: idx + 1 for idx, pair in enumerate(item_count["like"].most_common())},
            "addcart": {pair[0]: idx + 1 for idx, pair in enumerate(item_count["addcart"].most_common())},
            "order": {pair[0]: idx + 1 for idx, pair in enumerate(item_count["order"].most_common())}
        }

        brand_rank = {
            "click": {pair[0]: idx + 1 for idx, pair in enumerate(brand_count["click"].most_common())},
            "like": {pair[0]: idx + 1 for idx, pair in enumerate(brand_count["like"].most_common())},
            "addcart": {pair[0]: idx + 1 for idx, pair in enumerate(brand_count["addcart"].most_common())},
            "order": {pair[0]: idx + 1 for idx, pair in enumerate(brand_count["order"].most_common())}
        }

        category_rank = {
            "click": {pair[0]: idx + 1 for idx, pair in enumerate(category_count["click"].most_common())},
            "like": {pair[0]: idx + 1 for idx, pair in enumerate(category_count["like"].most_common())},
            "addcart": {pair[0]: idx + 1 for idx, pair in enumerate(category_count["addcart"].most_common())},
            "order": {pair[0]: idx + 1 for idx, pair in enumerate(category_count["order"].most_common())}
        }

        curr_item, curr_brand, curr_category = row["goods_id"], row["brand_id"], row["category_id"]

        features["instance_id"].append(row["instance_id"])

        features["item_click"].append(item_rank["click"].get(curr_item, 0))
        features["item_like"].append(item_rank["like"].get(curr_item, 0))
        features["item_addcart"].append(item_rank["addcart"].get(curr_item, 0))
        features["item_order"].append(item_rank["order"].get(curr_item, 0))

        features["brand_click"].append(brand_rank["click"].get(curr_brand, 0))
        features["brand_like"].append(brand_rank["like"].get(curr_brand, 0))
        features["brand_addcart"].append(brand_rank["addcart"].get(curr_brand, 0))
        features["brand_order"].append(brand_rank["order"].get(curr_brand, 0))

        features["category_click"].append(category_rank["click"].get(curr_category, 0))
        features["category_like"].append(category_rank["like"].get(curr_category, 0))
        features["category_addcart"].append(category_rank["addcart"].get(curr_category, 0))
        features["category_order"].append(category_rank["order"].get(curr_category, 0))

    return pd.DataFrame(features)


if __name__ == "__main__":
    run_multi_pool(get_top_user_behavior_sequence, file_name="top_user_behavior_sequence")
