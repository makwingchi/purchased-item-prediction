import pandas as pd

from multi_pool import run_multi_pool


def get_counts_and_rates(i):
    training = pd.read_csv("./训练集/training_data.csv")
    training["timestamp"] = pd.to_datetime(training["timestamp"], format="%Y-%m-%d %H:%M:%S")
    training["dt"] = pd.to_datetime(training["dt"], format="%Y%m%d")

    data = pd.read_csv(f"./训练集/users/part_{i:02d}.csv")
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y-%m-%d %H:%M:%S")
    data["dt"] = pd.to_datetime(data["dt"], format="%Y-%m-%d")

    features = {
        "instance_id": [],
        "before_user_click": [],
        "before_user_like": [],
        "before_user_addcart": [],
        "before_user_order": [],
        "before_item_click": [],
        "before_item_like": [],
        "before_item_addcart": [],
        "before_item_order": [],
        "before_brand_click": [],
        "before_brand_like": [],
        "before_brand_addcart": [],
        "before_brand_order": [],
        "before_category_click": [],
        "before_category_like": [],
        "before_category_addcart": [],
        "before_category_order": [],
        "before_user_like_rate": [],
        "before_user_addcart_rate": [],
        "before_user_order_rate": [],
        "before_item_like_rate": [],
        "before_item_addcart_rate": [],
        "before_item_order_rate": [],
        "before_brand_like_rate": [],
        "before_brand_addcart_rate": [],
        "before_brand_order_rate": [],
        "before_category_like_rate": [],
        "before_category_addcart_rate": [],
        "before_category_order_rate": [],
    }

    for idx, row in data.iterrows():
        before = training[training["timestamp"].lt(row["timestamp"])].reset_index(drop=True)

        before_user_click = before[before["user_id"].eq(row["user_id"]), "is_click"].sum()
        before_user_like = before[before["user_id"].eq(row["user_id"]), "is_like"].sum()
        before_user_addcart = before[before["user_id"].eq(row["user_id"]), "is_addcart"].sum()
        before_user_order = before[before["user_id"].eq(row["user_id"]), "is_order"].sum()

        before_item_click = before[before["goods_id"].eq(row["goods_id"]), "is_click"].sum()
        before_item_like = before[before["goods_id"].eq(row["goods_id"]), "is_like"].sum()
        before_item_addcart = before[before["goods_id"].eq(row["goods_id"]), "is_addcart"].sum()
        before_item_order = before[before["goods_id"].eq(row["goods_id"]), "is_order"].sum()

        before_brand_click = before[before["brand_id"].eq(row["brand_id"]), "is_click"].sum()
        before_brand_like = before[before["brand_id"].eq(row["brand_id"]), "is_like"].sum()
        before_brand_addcart = before[before["brand_id"].eq(row["brand_id"]), "is_addcart"].sum()
        before_brand_order = before[before["brand_id"].eq(row["brand_id"]), "is_order"].sum()

        before_category_click = before[before["category_id"].eq(row["category_id"]), "is_click"].sum()
        before_category_like = before[before["category_id"].eq(row["category_id"]), "is_like"].sum()
        before_category_addcart = before[before["category_id"].eq(row["category_id"]), "is_addcart"].sum()
        before_category_order = before[before["category_id"].eq(row["category_id"]), "is_order"].sum()

        before_user_like_rate = before_user_like / before_user_click
        before_user_addcart_rate = before_user_addcart / before_user_click
        before_user_order_rate = before_user_order / before_user_click

        before_item_like_rate = before_item_like / before_item_click
        before_item_addcart_rate = before_item_addcart / before_item_click
        before_item_order_rate = before_item_order / before_item_click

        before_brand_like_rate = before_brand_like / before_brand_click
        before_brand_addcart_rate = before_brand_addcart / before_brand_click
        before_brand_order_rate = before_brand_order / before_brand_click

        before_category_like_rate = before_category_like / before_category_click
        before_category_addcart_rate = before_category_addcart / before_category_click
        before_category_order_rate = before_category_order / before_category_click

        features["instance_id"].append(row["instance_id"])

        features["before_user_click"].append(before_user_click)
        features["before_user_like"].append(before_user_like)
        features["before_user_addcart"].append(before_user_addcart)
        features["before_user_order"].append(before_user_order)
        features["before_item_click"].append(before_item_click)
        features["before_item_like"].append(before_item_like)
        features["before_item_addcart"].append(before_item_addcart)
        features["before_item_order"].append(before_item_order)
        features["before_brand_click"].append(before_brand_click)
        features["before_brand_like"].append(before_brand_like)
        features["before_brand_addcart"].append(before_brand_addcart)
        features["before_brand_order"].append(before_brand_order)
        features["before_category_click"].append(before_category_click)
        features["before_category_like"].append(before_category_like)
        features["before_category_addcart"].append(before_category_addcart)
        features["before_category_order"].append(before_category_order)
        features["before_user_like_rate"].append(before_user_like_rate)
        features["before_user_addcart_rate"].append(before_user_addcart_rate)
        features["before_user_order_rate"].append(before_user_order_rate)
        features["before_item_like_rate"].append(before_item_like_rate)
        features["before_item_addcart_rate"].append(before_item_addcart_rate)
        features["before_item_order_rate"].append(before_item_order_rate)
        features["before_brand_like_rate"].append(before_brand_like_rate)
        features["before_brand_addcart_rate"].append(before_brand_addcart_rate)
        features["before_brand_order_rate"].append(before_brand_order_rate)
        features["before_category_like_rate"].append(before_category_like_rate)
        features["before_category_addcart_rate"].append(before_category_addcart_rate)
        features["before_category_order_rate"].append(before_category_order_rate)

    return pd.DataFrame(features)


if __name__ == "__main__":
    run_multi_pool(get_counts_and_rates, file_name="counts_and_rates")
