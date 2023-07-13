def get_configurations():
    config = {
        "runner": {
            "seed": 666,
            "device": "gpu",
            "batch_size": 2*512,
            "train_epochs": 1,
            "infer_start_epoch": 0,
            "infer_end_epoch": 1,
            "thread_num": 1,
            "print_interval": 50,
            "max_len": 20,
            "train_goods_path": "../../训练集/traindata_goodsid/",
            "train_users_path": "../../训练集/traindata_user/",

            "test_goods_path": "../../测试集a/predict_goods_id/",
            "test_user_path": "../../测试集a/a榜需要预测的uid_5000.xlsx",

            "processed_train": "./process_train/",
            "processed_test": "./process_test/",
            "model_type": "baseline",

            "sampling_rate": 0.2,
            "model_save_path": "./output"
        },
        "optimizer": {
            "learning_rate": 0.001
        },
        "model": {
            "baseline": {
                "num_users": 51602,
                "num_goods": 3465659,
                "sparse_feature_dim": 256,
                "sparse_feature_number": 1,
                "dense_feature_dim": 1,
                "num_fields": 2,
                "fc_size": [512, 256, 64, 32],
                "activate": "relu"
            },
            "bitower": {},
            'tritower': {}, }
    }
    return config
