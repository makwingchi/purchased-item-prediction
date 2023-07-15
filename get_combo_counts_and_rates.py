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

    }

    for idx, row in data.iterrows():
        pass


if __name__ == "__main__":
    run_multi_pool(get_counts_and_rates, file_name="counts_and_rates_combo")