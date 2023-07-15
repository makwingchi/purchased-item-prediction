import pandas as pd

from multi_pool import run_multi_pool


def count_max_no(rows):
    max_no = 0
    res = []

    for row in rows:
        if row == 0:
            max_no += 1
            res.append(max_no)
        else:
            max_no = 0

    return 0 if len(res) == 0 else max(res)


def get_user_continuous_behavior(i):
    data = pd.read_csv(f"./训练集/users/part_{i:02d}.csv")
    data["timestamp"] = pd.to_datetime(data["timestamp"], format="%Y-%m-%d %H:%M:%S")
    data["dt"] = pd.to_datetime(data["dt"], format="%Y-%m-%d")

    data.sort_values(["user_id", "timestamp"], inplace=True, ignore_index=True)

    features = {
        "instance_id": [],
        "max_no_like": [],
        "max_no_addcart": [],
        "max_no_order": []
    }

    for idx, row in data.iterrows():
        features["instance_id"].append(row["instance_id"])

        before = data[data["timestamp"].lt(row["timestamp"]) & data["user_id"].eq(row["user_id"])]

        if before.shape[0] == 0:
            features["max_no_like"].append(0)
            features["max_no_addcart"].append(0)
            features["max_no_order"].append(0)

            continue

        counts = (
            before
            .groupby("user_id", as_index=False)
            .agg(
                {
                    "is_like": lambda x: count_max_no(x),
                    "is_addcart": lambda x: count_max_no(x),
                    "is_order": lambda x: count_max_no(x),
                }
            )
        )

        features["max_no_like"].append(counts["is_like"].item())
        features["max_no_addcart"].append(counts["is_addcart"].item())
        features["max_no_order"].append(counts["is_order"].item())

    return pd.DataFrame(features)


if __name__ == "__main__":
    run_multi_pool(get_user_continuous_behavior, file_name="user_continuous_behavior")
