import math
from multiprocessing import cpu_count

import pandas as pd


def split_data(data, field, prefix, processor=cpu_count() - 2):
    data = data.sort_values(by=[field, 'timestamp']).reset_index(drop=True)

    curr_field = pd.DataFrame(list(set(data[field].values)), columns=[field])
    curr_field = curr_field.sample(frac=1).reset_index(drop=True)

    l_data = len(curr_field)
    size = math.ceil(l_data / processor)

    for i in range(processor):
        start = size * i
        end = (i + 1) * size if (i + 1) * size < l_data else l_data
        curr = curr_field[start:end]
        t_data = pd.merge(data, curr, on=field).reset_index(drop=True)

        t_data.to_csv(f'./训练集/{prefix}/part_{i:02d}.csv', index=False)
        print(len(t_data))


if __name__ == "__main__":
    training = pd.read_csv("../训练集/training_data.csv")
    training["timestamp"] = pd.to_datetime(training["timestamp"], format="%Y-%m-%d %H:%M:%S")
    training["dt"] = pd.to_datetime(training["dt"], format="%Y%m%d")

    training["instance_id"] = training.index
    training.sort_values(["user_id", "timestamp"], ignore_index=True, inplace=True)

    split_data(training, field="user_id", prefix="users")
