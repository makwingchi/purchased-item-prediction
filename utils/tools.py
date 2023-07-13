from sklearn.preprocessing import LabelEncoder
#
#
# def create_data_loader(config, place, task, mode="train"):
#     if mode == "train":
#         data_dir = config["runner"]["train_data_path"]
#         batch_size = config["runner"]["batch_size"]
#     else:
#         data_dir = config["runner"]["test_data_path"]
#         batch_size = config["runner"]["batch_size"]
#
#
#
#     return loader

def f1_score(precision, recall):
    return 2*precision*recall/(precision+recall)

def encode(user, good):
    user_encode = LabelEncoder()
    user_encode.fit(user)

    goods_encode = LabelEncoder()
    goods_encode.fit(good)

    return user_encode, goods_encode