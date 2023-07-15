import numpy as np
import pandas as pd

from multi_pool import run_multi_pool


def get_top_user_behavior_sequence(i):
    data = pd.read_csv(f"./训练集/users/part_{i:02d}.csv")
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y-%m-%d %H:%M:%S")
    data["dt"] = pd.to_datetime(data["dt"], format="%Y-%m-%d")

    features = {
        "instance_id": [],
        "same_item_click": [],
        "same_item_like": [],
        "same_item_addcart": [],
        "same_item_order": [],
        "same_brand_click": [],
        "same_brand_like": [],
        "same_brand_addcart": [],
        "same_brand_order": [],
        "same_category_click": [],
        "same_category_like": [],
        "same_category_addcart": [],
        "same_category_order": [],
        "diff_item_click": [],
        "diff_item_like": [],
        "diff_item_addcart": [],
        "diff_item_order": [],
        "diff_brand_click": [],
        "diff_brand_like": [],
        "diff_brand_addcart": [],
        "diff_brand_order": [],
        "diff_category_click": [],
        "diff_category_like": [],
        "diff_category_addcart": [],
        "diff_category_order": [],
        "before_min_item_click": [],
        "before_min_item_like": [],
        "before_min_item_addcart": [],
        "before_min_item_order": [],
        "before_min_brand_click": [],
        "before_min_brand_like": [],
        "before_min_brand_addcart": [],
        "before_min_brand_order": [],
        "before_min_category_click": [],
        "before_min_category_like": [],
        "before_min_category_addcart": [],
        "before_min_category_order": []
    }

    for idx, row in data.iterrows():
        tmp = data[data.user_id.eq(row["user_id"])]

        same_item_click = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["goods_id"].eq(row["goods_id"]) & tmp["is_click"].gt(0)])
        same_item_like = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["goods_id"].eq(row["goods_id"]) & tmp["is_like"].gt(0)])
        same_item_addcart = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["goods_id"].eq(row["goods_id"]) & tmp["is_addcart"].gt(0)])
        same_item_order = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["goods_id"].eq(row["goods_id"]) & tmp["is_order"].gt(0)])

        same_brand_click = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["brand_id"].eq(row["brand_id"]) & tmp["is_click"].gt(0)])
        same_brand_like = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["brand_id"].eq(row["brand_id"]) & tmp["is_like"].gt(0)])
        same_brand_addcart = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["brand_id"].eq(row["brand_id"]) & tmp["is_addcart"].gt(0)])
        same_brand_order = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["brand_id"].eq(row["brand_id"]) & tmp["is_order"].gt(0)])

        same_category_click = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["category_id"].eq(row["category_id"]) & tmp["is_click"].gt(0)])
        same_category_like = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["category_id"].eq(row["category_id"]) & tmp["is_like"].gt(0)])
        same_category_addcart = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["category_id"].eq(row["category_id"]) & tmp["is_addcart"].gt(0)])
        same_category_order = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["category_id"].eq(row["category_id"]) & tmp["is_order"].gt(0)])

        diff_item_click = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["goods_id"].ne(row["goods_id"]) & tmp["is_click"].gt(0)])
        diff_item_like = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["goods_id"].ne(row["goods_id"]) & tmp["is_like"].gt(0)])
        diff_item_addcart = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["goods_id"].ne(row["goods_id"]) & tmp["is_addcart"].gt(0)])
        diff_item_order = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["goods_id"].ne(row["goods_id"]) & tmp["is_order"].gt(0)])

        diff_brand_click = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["brand_id"].ne(row["brand_id"]) & tmp["is_click"].gt(0)])
        diff_brand_like = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["brand_id"].ne(row["brand_id"]) & tmp["is_like"].gt(0)])
        diff_brand_addcart = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["brand_id"].ne(row["brand_id"]) & tmp["is_addcart"].gt(0)])
        diff_brand_order = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["brand_id"].ne(row["brand_id"]) & tmp["is_order"].gt(0)])

        diff_category_click = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["category_id"].ne(row["category_id"]) & tmp["is_click"].gt(0)])
        diff_category_like = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["category_id"].ne(row["category_id"]) & tmp["is_like"].gt(0)])
        diff_category_addcart = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["category_id"].ne(row["category_id"]) & tmp["is_addcart"].gt(0)])
        diff_category_order = len(tmp[tmp["timestamp"].lt(row["timestamp"]) & tmp["category_id"].ne(row["category_id"]) & tmp["is_order"].gt(0)])

        item_min_time = np.min(tmp.loc[tmp["goods_id"].eq(row["goods_id"]), "timestamp"])
        brand_min_time = np.min(tmp.loc[tmp["brand_id"].eq(row["brand_id"]), "timestamp"])
        category_min_time = np.min(tmp.loc[tmp["category_id"].eq(row["category_id"]), "timestamp"])

        before_min_item_click = tmp.loc[tmp["timestamp"].lt(item_min_time) & tmp["is_click"].gt(0), "goods_id"].nunique()
        before_min_item_like = tmp.loc[tmp["timestamp"].lt(item_min_time) & tmp["is_like"].gt(0), "goods_id"].nunique()
        before_min_item_addcart = tmp.loc[tmp["timestamp"].lt(item_min_time) & tmp["is_addcart"].gt(0), "goods_id"].nunique()
        before_min_item_order = tmp.loc[tmp["timestamp"].lt(item_min_time) & tmp["is_order"].gt(0), "goods_id"].nunique()

        before_min_brand_click = tmp.loc[tmp["timestamp"].lt(brand_min_time) & tmp["is_click"].gt(0), "brand_id"].nunique()
        before_min_brand_like = tmp.loc[tmp["timestamp"].lt(brand_min_time) & tmp["is_like"].gt(0), "brand_id"].nunique()
        before_min_brand_addcart = tmp.loc[tmp["timestamp"].lt(brand_min_time) & tmp["is_addcart"].gt(0), "brand_id"].nunique()
        before_min_brand_order = tmp.loc[tmp["timestamp"].lt(brand_min_time) & tmp["is_order"].gt(0), "brand_id"].nunique()

        before_min_category_click = tmp.loc[tmp["timestamp"].lt(category_min_time) & tmp["is_click"].gt(0), "category_id"].nunique()
        before_min_category_like = tmp.loc[tmp["timestamp"].lt(category_min_time) & tmp["is_like"].gt(0), "category_id"].nunique()
        before_min_category_addcart = tmp.loc[tmp["timestamp"].lt(category_min_time) & tmp["is_addcart"].gt(0), "category_id"].nunique()
        before_min_category_order = tmp.loc[tmp["timestamp"].lt(category_min_time) & tmp["is_order"].gt(0), "category_id"].nunique()

        features["instance_id"].append(row["instance_id"])

        features["same_item_click"].append(same_item_click)
        features["same_item_like"].append(same_item_like)
        features["same_item_addcart"].append(same_item_addcart)
        features["same_item_order"].append(same_item_order)

        features["same_brand_click"].append(same_brand_click)
        features["same_brand_like"].append(same_brand_like)
        features["same_brand_addcart"].append(same_brand_addcart)
        features["same_brand_order"].append(same_brand_order)

        features["same_category_click"].append(same_category_click)
        features["same_category_like"].append(same_category_like)
        features["same_category_addcart"].append(same_category_addcart)
        features["same_category_order"].append(same_category_order)

        features["diff_item_click"].append(diff_item_click)
        features["diff_item_like"].append(diff_item_like)
        features["diff_item_addcart"].append(diff_item_addcart)
        features["diff_item_order"].append(diff_item_order)

        features["diff_brand_click"].append(diff_brand_click)
        features["diff_brand_like"].append(diff_brand_like)
        features["diff_brand_addcart"].append(diff_brand_addcart)
        features["diff_brand_order"].append(diff_brand_order)

        features["diff_category_click"].append(diff_category_click)
        features["diff_category_like"].append(diff_category_like)
        features["diff_category_addcart"].append(diff_category_addcart)
        features["diff_category_order"].append(diff_category_order)

        features["before_min_item_click"].append(before_min_item_click)
        features["before_min_item_like"].append(before_min_item_like)
        features["before_min_item_addcart"].append(before_min_item_addcart)
        features["before_min_item_order"].append(before_min_item_order)

        features["before_min_brand_click"].append(before_min_brand_click)
        features["before_min_brand_like"].append(before_min_brand_like)
        features["before_min_brand_addcart"].append(before_min_brand_addcart)
        features["before_min_brand_order"].append(before_min_brand_order)

        features["before_min_category_click"].append(before_min_category_click)
        features["before_min_category_like"].append(before_min_category_like)
        features["before_min_category_addcart"].append(before_min_category_addcart)
        features["before_min_category_order"].append(before_min_category_order)

    return pd.DataFrame(features)


if __name__ == "__main__":
    run_multi_pool(get_top_user_behavior_sequence, file_name="item_brand_category_count")
