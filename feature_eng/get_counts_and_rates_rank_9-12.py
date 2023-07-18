import datetime

import pandas as pd

from multi_pool import run_multi_pool


def calculate(before, field):
    curr = (
        before
        .groupby(field, as_index=False)
        .agg({"is_click": "sum", "is_like": "sum", "is_addcart": "sum", "is_order": "sum"})
    )
    curr["like_rate"] = curr["is_like"] / curr["is_click"]
    curr["addcart_rate"] = curr["is_addcart"] / curr["is_click"]
    curr["order_rate"] = curr["is_order"] / curr["is_click"]

    curr["click_rank"] = curr['is_click'].rank(method='dense', ascending=False)
    curr["like_rank"] = curr['is_like'].rank(method='dense', ascending=False)
    curr["addcart_rank"] = curr['is_addcart'].rank(method='dense', ascending=False)
    curr["order_rank"] = curr['is_order'].rank(method='dense', ascending=False)
    curr["like_rate_rank"] = curr['like_rate'].rank(method='dense', ascending=False)
    curr["addcart_rate_rank"] = curr['addcart_rate'].rank(method='dense', ascending=False)
    curr["order_rate_rank"] = curr['order_rate'].rank(method='dense', ascending=False)

    return curr


"""
Features 9-12 (removed)
"""

FILE_NAME = "counts_and_rates_rank"


def get_counts_and_rates_rank(i):
    training = pd.read_csv("./训练集/training_data.csv")
    training["timestamp"] = pd.to_datetime(training["timestamp"], format="%Y-%m-%d %H:%M:%S")

    training["timestamp"] = training["timestamp"].astype(int) // 10 ** 9

    data = pd.read_csv(f"./训练集/users/part_{i:02d}.csv")
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y-%m-%d %H:%M:%S")

    data["timestamp"] = data["timestamp"].astype(int) // 10 ** 9

    features = {
        "instance_id": [],
        "user_click_rank": [],
        "user_like_rank": [],
        "user_addcart_rank": [],
        "user_order_rank": [],
        "user_like_rate_rank": [],
        "user_addcart_rate_rank": [],
        "user_order_rate_rank": [],
        "item_click_rank": [],
        "item_like_rank": [],
        "item_addcart_rank": [],
        "item_order_rank": [],
        "item_like_rate_rank": [],
        "item_addcart_rate_rank": [],
        "item_order_rate_rank": [],
        "brand_click_rank": [],
        "brand_like_rank": [],
        "brand_addcart_rank": [],
        "brand_order_rank": [],
        "brand_like_rate_rank": [],
        "brand_addcart_rate_rank": [],
        "brand_order_rate_rank": [],
        "category_click_rank": [],
        "category_like_rank": [],
        "category_addcart_rank": [],
        "category_order_rank": [],
        "category_like_rate_rank": [],
        "category_addcart_rate_rank": [],
        "category_order_rate_rank": [],
    }

    timestamp_arr = training["timestamp"].to_numpy()

    for idx, row in data.iterrows():
        if idx % 50 == 0:
            print(idx)
            print(datetime.datetime.now())

        indices = timestamp_arr < row["timestamp"]
        before = training.loc[indices, :].reset_index(drop=True)

        before_user = calculate(before, "user_id")
        before_item = calculate(before, "goods_id")
        before_brand = calculate(before, "brand_id")
        before_category = calculate(before, "category_id")

        curr_user, curr_item, curr_brand, curr_category = row["user_id"], row["goods_id"], row["brand_id"], row["category_id"]

        curr_user_ranks = before_user.loc[before_user["user_id"].eq(curr_user)]
        user_click_rank = curr_user_ranks["click_rank"].item() if curr_user_ranks.shape[0] > 0 else 0
        user_like_rank = curr_user_ranks["like_rank"].item() if curr_user_ranks.shape[0] > 0 else 0
        user_addcart_rank = curr_user_ranks["addcart_rank"].item() if curr_user_ranks.shape[0] > 0 else 0
        user_order_rank = curr_user_ranks["order_rank"].item() if curr_user_ranks.shape[0] > 0 else 0
        user_like_rate_rank = curr_user_ranks["like_rate_rank"].item() if curr_user_ranks.shape[0] > 0 else 0
        user_addcart_rate_rank = curr_user_ranks["addcart_rate_rank"].item() if curr_user_ranks.shape[0] > 0 else 0
        user_order_rate_rank = curr_user_ranks["order_rate_rank"].item() if curr_user_ranks.shape[0] > 0 else 0

        curr_item_ranks = before_item.loc[before_item["goods_id"].eq(curr_item)]
        item_click_rank = curr_item_ranks["click_rank"].item() if curr_item_ranks.shape[0] > 0 else 0
        item_like_rank = curr_item_ranks["like_rank"].item() if curr_item_ranks.shape[0] > 0 else 0
        item_addcart_rank = curr_item_ranks["addcart_rank"].item() if curr_item_ranks.shape[0] > 0 else 0
        item_order_rank = curr_item_ranks["order_rank"].item() if curr_item_ranks.shape[0] > 0 else 0
        item_like_rate_rank = curr_item_ranks["like_rate_rank"].item() if curr_item_ranks.shape[0] > 0 else 0
        item_addcart_rate_rank = curr_item_ranks["addcart_rate_rank"].item() if curr_item_ranks.shape[0] > 0 else 0
        item_order_rate_rank = curr_item_ranks["order_rate_rank"].item() if curr_item_ranks.shape[0] > 0 else 0

        curr_brand_ranks = before_brand.loc[before_brand["brand_id"].eq(curr_brand)]
        brand_click_rank = curr_brand_ranks["click_rank"].item() if curr_brand_ranks.shape[0] > 0 else 0
        brand_like_rank = curr_brand_ranks["like_rank"].item() if curr_brand_ranks.shape[0] > 0 else 0
        brand_addcart_rank = curr_brand_ranks["addcart_rank"].item() if curr_brand_ranks.shape[0] > 0 else 0
        brand_order_rank = curr_brand_ranks["order_rank"].item() if curr_brand_ranks.shape[0] > 0 else 0
        brand_like_rate_rank = curr_brand_ranks["like_rate_rank"].item() if curr_brand_ranks.shape[0] > 0 else 0
        brand_addcart_rate_rank = curr_brand_ranks["addcart_rate_rank"].item() if curr_brand_ranks.shape[0] > 0 else 0
        brand_order_rate_rank = curr_brand_ranks["order_rate_rank"].item() if curr_brand_ranks.shape[0] > 0 else 0

        curr_category_ranks = before_category.loc[before_category["category_id"].eq(curr_category)]
        category_click_rank = curr_category_ranks["click_rank"].item() if curr_category_ranks.shape[0] > 0 else 0
        category_like_rank = curr_category_ranks["like_rank"].item() if curr_category_ranks.shape[0] > 0 else 0
        category_addcart_rank = curr_category_ranks["addcart_rank"].item() if curr_category_ranks.shape[0] > 0 else 0
        category_order_rank = curr_category_ranks["order_rank"].item() if curr_category_ranks.shape[0] > 0 else 0
        category_like_rate_rank = curr_category_ranks["like_rate_rank"].item() if curr_category_ranks.shape[0] > 0 else 0
        category_addcart_rate_rank = curr_category_ranks["addcart_rate_rank"].item() if curr_category_ranks.shape[0] > 0 else 0
        category_order_rate_rank = curr_category_ranks["order_rate_rank"].item() if curr_category_ranks.shape[0] > 0 else 0

        features["instance_id"].append(row["instance_id"])

        features["user_click_rank"].append(user_click_rank)
        features["user_like_rank"].append(user_like_rank)
        features["user_addcart_rank"].append(user_addcart_rank)
        features["user_order_rank"].append(user_order_rank)
        features["user_like_rate_rank"].append(user_like_rate_rank)
        features["user_addcart_rate_rank"].append(user_addcart_rate_rank)
        features["user_order_rate_rank"].append(user_order_rate_rank)

        features["item_click_rank"].append(item_click_rank)
        features["item_like_rank"].append(item_like_rank)
        features["item_addcart_rank"].append(item_addcart_rank)
        features["item_order_rank"].append(item_order_rank)
        features["item_like_rate_rank"].append(item_like_rate_rank)
        features["item_addcart_rate_rank"].append(item_addcart_rate_rank)
        features["item_order_rate_rank"].append(item_order_rate_rank)

        features["brand_click_rank"].append(brand_click_rank)
        features["brand_like_rank"].append(brand_like_rank)
        features["brand_addcart_rank"].append(brand_addcart_rank)
        features["brand_order_rank"].append(brand_order_rank)
        features["brand_like_rate_rank"].append(brand_like_rate_rank)
        features["brand_addcart_rate_rank"].append(brand_addcart_rate_rank)
        features["brand_order_rate_rank"].append(brand_order_rate_rank)

        features["category_click_rank"].append(category_click_rank)
        features["category_like_rank"].append(category_like_rank)
        features["category_addcart_rank"].append(category_addcart_rank)
        features["category_order_rank"].append(category_order_rank)
        features["category_like_rate_rank"].append(category_like_rate_rank)
        features["category_addcart_rate_rank"].append(category_addcart_rate_rank)
        features["category_order_rate_rank"].append(category_order_rate_rank)

    print(f"finish processing part_{i:02d}")
    pd.DataFrame(features).to_csv(f"./训练集/{FILE_NAME}/part_{i:02d}.csv", index=False)
    print(f"part_{i:02d} saved")

    return pd.DataFrame(features)


if __name__ == "__main__":
    run_multi_pool(get_counts_and_rates_rank, file_name=FILE_NAME)
    # get_counts_and_rates_rank(0)
