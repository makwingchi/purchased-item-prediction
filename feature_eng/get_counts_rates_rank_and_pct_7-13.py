import datetime

import joblib
import pandas as pd


"""
Features 7-13 (half-day split)
"""


FILE_NAME = "counts_rates_rank_and_pct"


def get_closest_timestamp_ahead(timestamp):
    dt = timestamp

    if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
        return dt + datetime.timedelta(hours=-24)
    else:
        return dt.replace(hour=0, minute=0, second=0)


def fill_in_features(features, curr_dict, timestamp, curr_key, fields, field_name, prefix=True):
    if timestamp not in curr_dict:
        for field in fields:
            if prefix:
                new_field_name = field_name + "_" + field
            else:
                new_field_name = field

            if new_field_name not in features:
                features[new_field_name] = []

            features[new_field_name].append(0)

        return

    values = curr_dict[timestamp][curr_key]

    for field in fields:
        if field not in values:
            continue

        if prefix:
            new_field_name = field_name + "_" + field
        else:
            new_field_name = field

        if new_field_name not in features:
            features[new_field_name] = []

        features[new_field_name].append(values[field])


def get_counts_rates_rank_and_pct(ii, user_dict, brand_dict, category_dict):
    data = pd.read_csv(f"./训练集/users/part_{ii:02d}.csv")
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y-%m-%d %H:%M:%S")

    features = {"instance_id": []}

    for idx, row in data.iterrows():
        if idx % 5000 == 0:
            print(idx)
            print(datetime.datetime.now())

        features["instance_id"].append(row["instance_id"])

        curr_timestamp = row["timestamp"]
        closest_timestamp = get_closest_timestamp_ahead(curr_timestamp)

        curr_user = str(row["user_id"])
        # curr_item = str(row["goods_id"])
        curr_brand = str(row["brand_id"])
        curr_category = str(row["category_id"])

        # curr_user_item = curr_user + "_" + curr_item
        # curr_user_brand = curr_user + "_" + curr_brand
        # curr_user_category = curr_user + "_" + curr_category
        # curr_brand_category = curr_brand + "_" + curr_category

        fields = [
            "is_click", "is_like", "is_addcart", "is_order", "like_rate", "addcart_rate", "order_rate", "click_rank",
            "like_rank", "addcart_rank", "order_rank", "like_rate_rank", "addcart_rate_rank", "order_rate_rank"
        ]

        fill_in_features(features, user_dict, closest_timestamp, curr_user, fields, field_name="user")
        # fill_in_features(features, item_dict, closest_timestamp, curr_item, fields, field_name="item")
        fill_in_features(features, brand_dict, closest_timestamp, curr_brand, fields, field_name="brand")
        fill_in_features(features, category_dict, closest_timestamp, curr_category, fields, field_name="category")

        # fields += [
        #     "user_item_user_click", "user_item_user_like", "user_item_user_addcart", "user_item_user_order",
        #     "user_item_item_click", "user_item_item_like", "user_item_item_addcart", "user_item_item_order",
        #     "user_brand_user_click", "user_brand_user_like", "user_brand_user_addcart", "user_brand_user_order",
        #     "user_brand_brand_click", "user_brand_brand_like", "user_brand_brand_addcart", "user_brand_brand_order",
        #     "user_category_user_click", "user_category_user_like", "user_category_user_addcart", "user_category_user_order",
        #     "user_category_category_click", "user_category_category_like", "user_category_category_addcart", "user_category_category_order",
        #     "brand_category_brand_click", "brand_category_brand_like", "brand_category_brand_addcart", "brand_category_brand_order",
        #     "brand_category_category_click", "brand_category_category_like", "brand_category_category_addcart", "brand_category_category_order"
        # ]
        #
        # fill_in_features(features, user_item_dict, closest_timestamp, curr_user_item, fields, None, False)
        # fill_in_features(features, user_brand_dict, closest_timestamp, curr_user_brand, fields, None, False)
        # fill_in_features(features, user_category_dict, closest_timestamp, curr_user_category, fields, None, False)
        # fill_in_features(features, brand_category_dict, closest_timestamp, curr_brand_category, fields, None, False)

    print(f"finish processing part_{ii:02d}")
    pd.DataFrame(features).to_csv(f"./训练集/{FILE_NAME}/part_{ii:02d}.csv", index=False)
    print(f"part_{ii:02d} saved")

    return pd.DataFrame(features)


if __name__ == "__main__":
    user_dict = joblib.load("./训练集/user_dict.pkl")
    brand_dict = joblib.load("./训练集/brand_dict.pkl")
    category_dict = joblib.load("./训练集/category_dict.pkl")

    for i in range(22):
        get_counts_rates_rank_and_pct(i, user_dict, brand_dict, category_dict)
