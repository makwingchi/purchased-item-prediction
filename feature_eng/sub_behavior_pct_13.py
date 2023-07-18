import datetime
import pandas as pd

"""
Feature 13 (integrated in get_counts_rates_and_pct_7-8_13)
"""

FILE_NAME = "sub_behavior_pct"


def get_sub_behavior_pct(i):
    training = pd.read_csv("../训练集/training_data.csv")
    training["timestamp"] = pd.to_datetime(training["timestamp"], format="%Y-%m-%d %H:%M:%S")

    training["timestamp"] = training["timestamp"].astype(int) // 10 ** 9

    data = pd.read_csv(f"./训练集/users/part_{i:02d}.csv")
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y-%m-%d %H:%M:%S")

    data["timestamp"] = data["timestamp"].astype(int) // 10 ** 9

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
            print(idx)
            print(datetime.datetime.now())

        before = timestamp_arr < row["timestamp"]

        curr_user = before & (user_arr == row["user_id"])
        curr_item = before & (item_arr == row["goods_id"])
        curr_brand = before & (brand_arr == row["brand_id"])
        curr_category = before & (category_arr == row["category_id"])

        user_click = click_arr[curr_user].sum()
        user_like = like_arr[curr_user].sum()
        user_addcart = addcart_arr[curr_user].sum()
        user_order = order_arr[curr_user].sum()

        item_click = click_arr[curr_item].sum()
        item_like = like_arr[curr_item].sum()
        item_addcart = addcart_arr[curr_item].sum()
        item_order = order_arr[curr_item].sum()

        brand_click = click_arr[curr_brand].sum()
        brand_like = like_arr[curr_brand].sum()
        brand_addcart = addcart_arr[curr_brand].sum()
        brand_order = order_arr[curr_brand].sum()

        category_click = click_arr[curr_category].sum()
        category_like = like_arr[curr_category].sum()
        category_addcart = addcart_arr[curr_category].sum()
        category_order = order_arr[curr_category].sum()

        user_item_click = click_arr[curr_user & curr_item].sum()
        user_item_like = like_arr[curr_user & curr_item].sum()
        user_item_addcart = addcart_arr[curr_user & curr_item].sum()
        user_item_order = order_arr[curr_user & curr_item].sum()

        user_brand_click = click_arr[curr_user & curr_brand].sum()
        user_brand_like = like_arr[curr_user & curr_brand].sum()
        user_brand_addcart = addcart_arr[curr_user & curr_brand].sum()
        user_brand_order = order_arr[curr_user & curr_brand].sum()

        user_category_click = click_arr[curr_user & curr_category].sum()
        user_category_like = like_arr[curr_user & curr_category].sum()
        user_category_addcart = addcart_arr[curr_user & curr_category].sum()
        user_category_order = order_arr[curr_user & curr_category].sum()

        brand_category_click = click_arr[curr_brand & curr_category].sum()
        brand_category_like = like_arr[curr_brand & curr_category].sum()
        brand_category_addcart = addcart_arr[curr_brand & curr_category].sum()
        brand_category_order = order_arr[curr_brand & curr_category].sum()

        user_item_user_click = user_item_click / (user_click + 3)
        user_item_user_like = user_item_like / (user_like + 3)
        user_item_user_addcart = user_item_addcart / (user_addcart + 3)
        user_item_user_order = user_item_order / (user_order + 3)

        user_item_item_click = user_item_click / (item_click + 3)
        user_item_item_like = user_item_like / (item_like + 3)
        user_item_item_addcart = user_item_addcart / (item_addcart + 3)
        user_item_item_order = user_item_order / (item_order + 3)

        user_brand_user_click = user_brand_click / (user_click + 3)
        user_brand_user_like = user_brand_like / (user_like + 3)
        user_brand_user_addcart = user_brand_addcart / (user_addcart + 3)
        user_brand_user_order = user_brand_order / (user_order + 3)

        user_brand_brand_click = user_brand_click / (brand_click + 3)
        user_brand_brand_like = user_brand_like / (brand_like + 3)
        user_brand_brand_addcart = user_brand_addcart / (brand_addcart + 3)
        user_brand_brand_order = user_brand_order / (brand_order + 3)

        user_category_user_click = user_category_click / (user_click + 3)
        user_category_user_like = user_category_like / (user_like + 3)
        user_category_user_addcart = user_category_addcart / (user_addcart + 3)
        user_category_user_order = user_category_order / (user_order + 3)

        user_category_category_click = user_category_click / (category_click + 3)
        user_category_category_like = user_category_like / (category_like + 3)
        user_category_category_addcart = user_category_addcart / (category_addcart + 3)
        user_category_category_order = user_category_order / (category_order + 3)

        brand_category_brand_click = brand_category_click / (brand_click + 3)
        brand_category_brand_like = brand_category_like / (brand_like + 3)
        brand_category_brand_addcart = brand_category_addcart / (brand_addcart + 3)
        brand_category_brand_order = brand_category_order / (brand_order+ 3)

        brand_category_category_click = brand_category_click / (category_click + 3)
        brand_category_category_like = brand_category_like / (category_like + 3)
        brand_category_category_addcart = brand_category_addcart / (category_addcart + 3)
        brand_category_category_order = brand_category_order / (category_order + 3)

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

    print(f"finish processing part_{i:02d}")
    pd.DataFrame(features).to_csv(f"./训练集/{FILE_NAME}/part_{i:02d}.csv", index=False)
    print(f"part_{i:02d} saved")

    return pd.DataFrame(features)


if __name__ == "__main__":
    # run_multi_pool(get_sub_behavior_pct, file_name=FILE_NAME)
    get_sub_behavior_pct(0)
