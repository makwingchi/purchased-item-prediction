import pandas as pd

from multi_pool import run_multi_pool

from get_counts_and_rates_rank import calculate


def calculate_twin(before, fields):
    curr = (
        before
        .groupby(fields, as_index=False)
        .agg({"is_click": "sum", "is_like": "sum", "is_addcart": "sum", "is_order": "sum"})
    )

    return curr


def get_sub_behavior_pct(i):
    training = pd.read_csv("./训练集/training_data.csv")
    training["timestamp"] = pd.to_datetime(training["timestamp"], format="%Y-%m-%d %H:%M:%S")
    training["dt"] = pd.to_datetime(training["dt"], format="%Y%m%d")

    data = pd.read_csv(f"./训练集/users/part_{i:02d}.csv")
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y-%m-%d %H:%M:%S")
    data["dt"] = pd.to_datetime(data["dt"], format="%Y-%m-%d")

    features = {
        "instance_id": [],
        "user_item_user_click": [],
        "user_item_user_like": [],
        "user_item_user_addcart": [],
        "user_item_user_order": [],
        "user_item_item_click": [],
        "user_item_item_like": [],
        "user_item_item_addcart": [],
        "user_item_item_order": [],
        "user_brand_user_click": [],
        "user_brand_user_like": [],
        "user_brand_user_addcart": [],
        "user_brand_user_order": [],
        "user_brand_brand_click": [],
        "user_brand_brand_like": [],
        "user_brand_brand_addcart": [],
        "user_brand_brand_order": [],
        "user_category_user_click": [],
        "user_category_user_like": [],
        "user_category_user_addcart": [],
        "user_category_user_order": [],
        "user_category_category_click": [],
        "user_category_category_like": [],
        "user_category_category_addcart": [],
        "user_category_category_order": [],
        "brand_category_brand_click": [],
        "brand_category_brand_like": [],
        "brand_category_brand_addcart": [],
        "brand_category_brand_order": [],
        "brand_category_category_click": [],
        "brand_category_category_like": [],
        "brand_category_category_addcart": [],
        "brand_category_category_order": [],
    }

    tmp_dict = {}

    for idx, row in data.iterrows():
        curr_timestamp = row["timestamp"]

        if str(curr_timestamp) in tmp_dict:
            before = tmp_dict[str(curr_timestamp)]
        else:
            before = training[training["timestamp"].lt(curr_timestamp)].reset_index(drop=True)
            tmp_dict[str(curr_timestamp)] = before

        before_user = calculate(before, "user_id")
        before_item = calculate(before, "goods_id")
        before_brand = calculate(before, "brand_id")
        before_category = calculate(before, "category_id")

        before_user_item = calculate_twin(before, fields=["user_id", "goods_id"])
        before_user_brand = calculate_twin(before, fields=["user_id", "goods_id"])
        before_user_category = calculate_twin(before, fields=["user_id", "goods_id"])
        before_brand_category = calculate_twin(before, fields=["user_id", "goods_id"])

        curr_user = before_user[before_user["user_id"].eq(row["user_id"])]
        curr_item = before_item[before_item["goods_id"].eq(row["goods_id"])]
        curr_brand = before_brand[before_brand["brand_id"].eq(row["brand_id"])]
        curr_category = before_category[before_category["category_id"].eq(row["category_id"])]

        curr_user_item = before_user_item[before_user_item["user_id"].eq(row["user_id"]) & before_user_item["goods_id"].eq(row["goods_id"])]
        curr_user_brand = before_user_brand[before_user_brand["user_id"].eq(row["user_id"]) & before_user_brand["brand_id"].eq(row["brand_id"])]
        curr_user_category = before_user_category[before_user_category["user_id"].eq(row["user_id"]) & before_user_category["category_id"].eq(row["category_id"])]
        curr_brand_category = before_brand_category[before_brand_category["brand_id"].eq(row["brand_id"]) & before_brand_category["category_id"].eq(row["category_id"])]

        user_item_user_click = curr_user_item["is_click"].sum() / (curr_user["is_click"].sum() + 3)
        user_item_user_like = curr_user_item["is_like"].sum() / (curr_user["is_like"].sum() + 3)
        user_item_user_addcart = curr_user_item["is_addcart"].sum() / (curr_user["is_addcart"].sum() + 3)
        user_item_user_order = curr_user_item["is_order"].sum() / (curr_user["is_order"].sum() + 3)

        user_item_item_click = curr_user_item["is_click"].sum() / (curr_item["is_click"].sum() + 3)
        user_item_item_like = curr_user_item["is_like"].sum() / (curr_item["is_like"].sum() + 3)
        user_item_item_addcart = curr_user_item["is_addcart"].sum() / (curr_item["is_addcart"].sum() + 3)
        user_item_item_order = curr_user_item["is_order"].sum() / (curr_item["is_order"].sum() + 3)

        user_brand_user_click = curr_user_brand["is_click"].sum() / (curr_user["is_click"].sum() + 3)
        user_brand_user_like = curr_user_brand["is_like"].sum() / (curr_user["is_like"].sum() + 3)
        user_brand_user_addcart = curr_user_brand["is_addcart"].sum() / (curr_user["is_addcart"].sum() + 3)
        user_brand_user_order = curr_user_brand["is_order"].sum() / (curr_user["is_order"].sum() + 3)

        user_brand_brand_click = curr_user_brand["is_click"].sum() / (curr_brand["is_click"].sum() + 3)
        user_brand_brand_like = curr_user_brand["is_like"].sum() / (curr_brand["is_like"].sum() + 3)
        user_brand_brand_addcart = curr_user_brand["is_addcart"].sum() / (curr_brand["is_addcart"].sum() + 3)
        user_brand_brand_order = curr_user_brand["is_order"].sum() / (curr_brand["is_order"].sum() + 3)

        user_category_user_click = curr_user_category["is_click"].sum() / (curr_user["is_click"].sum() + 3)
        user_category_user_like = curr_user_category["is_like"].sum() / (curr_user["is_like"].sum() + 3)
        user_category_user_addcart = curr_user_category["is_addcart"].sum() / (curr_user["is_addcart"].sum() + 3)
        user_category_user_order = curr_user_category["is_order"].sum() / (curr_user["is_order"].sum() + 3)
    
        user_category_category_click = curr_user_category["is_click"].sum() / (curr_category["is_click"].sum() + 3)
        user_category_category_like = curr_user_category["is_like"].sum() / (curr_category["is_like"].sum() + 3)
        user_category_category_addcart = curr_user_category["is_addcart"].sum() / (curr_category["is_addcart"].sum() + 3)
        user_category_category_order = curr_user_category["is_order"].sum() / (curr_category["is_order"].sum() + 3)
    
        brand_category_brand_click = curr_brand_category["is_click"].sum() / (curr_brand["is_click"].sum() + 3)
        brand_category_brand_like = curr_brand_category["is_like"].sum() / (curr_brand["is_like"].sum() + 3)
        brand_category_brand_addcart = curr_brand_category["is_addcart"].sum() / (curr_brand["is_addcart"].sum() + 3)
        brand_category_brand_order = curr_brand_category["is_order"].sum() / (curr_brand["is_order"].sum() + 3)
    
        brand_category_category_click = curr_brand_category["is_click"].sum() / (curr_category["is_click"].sum() + 3)
        brand_category_category_like = curr_brand_category["is_like"].sum() / (curr_category["is_like"].sum() + 3)
        brand_category_category_addcart = curr_brand_category["is_addcart"].sum() / (curr_category["is_addcart"].sum() + 3)
        brand_category_category_order = curr_brand_category["is_order"].sum() / (curr_category["is_order"].sum() + 3)

        features["instance_id"].append(row["instance_id"])

        features["user_item_user_click"].append(user_item_user_click)
        features["user_item_user_like"].append(user_item_user_like)
        features["user_item_user_addcart"].append(user_item_user_addcart)
        features["user_item_user_order"].append(user_item_user_order)

        features["user_item_item_click"].append(user_item_item_click)
        features["user_item_item_like"].append(user_item_item_like)
        features["user_item_item_addcart"].append(user_item_item_addcart)
        features["user_item_item_order"].append(user_item_item_order)

        features["user_brand_user_click"].append(user_brand_user_click)
        features["user_brand_user_like"].append(user_brand_user_like)
        features["user_brand_user_addcart"].append(user_brand_user_addcart)
        features["user_brand_user_order"].append(user_brand_user_order)

        features["user_brand_brand_click"].append(user_brand_brand_click)
        features["user_brand_brand_like"].append(user_brand_brand_like)
        features["user_brand_brand_addcart"].append(user_brand_brand_addcart)
        features["user_brand_brand_order"].append(user_brand_brand_order)

        features["user_category_user_click"].append(user_category_user_click)
        features["user_category_user_like"].append(user_category_user_like)
        features["user_category_user_addcart"].append(user_category_user_addcart)
        features["user_category_user_order"].append(user_category_user_order)

        features["user_category_category_click"].append(user_category_category_click)
        features["user_category_category_like"].append(user_category_category_like)
        features["user_category_category_addcart"].append(user_category_category_addcart)
        features["user_category_category_order"].append(user_category_category_order)

        features["brand_category_brand_click"].append(brand_category_brand_click)
        features["brand_category_brand_like"].append(brand_category_brand_like)
        features["brand_category_brand_addcart"].append(brand_category_brand_addcart)
        features["brand_category_brand_order"].append(brand_category_brand_order)

        features["brand_category_category_click"].append(brand_category_category_click)
        features["brand_category_category_like"].append(brand_category_category_like)
        features["brand_category_category_addcart"].append(brand_category_category_addcart)
        features["brand_category_category_order"].append(brand_category_category_order)

    return pd.DataFrame(features)


if __name__ == "__main__":
    run_multi_pool(get_sub_behavior_pct, file_name="sub_behavior_pct")
