import os
import pickle
import time
import logging
import argparse

import paddle
import pandas as pd
from utils.tools import f1_score, encode
from config import get_configurations
from utils.data_loader import get_batch_data
from utils.data_process import VIPDataProcessor
# from utils.utils_single import create_data_loader
from utils.save_and_load import save_model
from models.dynamic_model import DynamicModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt='[%Y-%m-%d %H:%M:%S]',
)
logger = logging.getLogger(__name__)

# parser = argparse.ArgumentParser()
# parser.add_argument("--conv_type", help="specify conversion type", default="1")
# parser.add_argument("--purpose", help="specify purpose", default="test")
# parser.add_argument("--model_type", help="specify model type", default="deepcrossing")
# parser.add_argument("--is_infer", help="specify mode", default="0")


if __name__ == "__main__":
    # args = parser.parse_args()
    # conv_type = str(args.conv_type)
    # purpose = str(args.purpose)
    # model_type = str(args.model_type)
    # is_infer = bool(int(args.is_infer))

    config = get_configurations()

    seed = config["runner"]["seed"]
    device = config["runner"]["device"]
    num_epochs = config["runner"]["train_epochs"]
    print_interval = config["runner"]["print_interval"]
    model_save_path = config["runner"]["model_save_path"]
    # task_type = config["runner"]["task_type"]
    processed_train_path = config['runner']['processed_train']
    processed_test_path = config['runner']['processed_test']
    sampling_rate = config['runner']['sampling_rate']

    user_ls, goods_ls = [], []
    if not os.path.exists(processed_train_path):
        logger.info('---------------->> genereate training set <<----------------')
        vdp = VIPDataProcessor(config, mode='train')
        train_data, user_ls, goods_ls = vdp.fe_process(save=True)
        logger.info('---------------->>     genereate end     <<----------------')
        del vdp
    else:
        logger.info('---------------->>   load training set   <<----------------')
        train_data = pd.read_csv(os.path.join(processed_train_path, 'data.csv')).reset_index(drop=True)

    if not os.path.exists(processed_test_path):
        logger.info('---------------->> genereate testing set <<----------------')
        vdp = VIPDataProcessor(config, mode='test')
        test_data = vdp.get_test_candidate(save=True)
        logger.info('---------------->>     genereate end     <<----------------')
        del vdp

    if not os.path.exists('./user_encoder.pkl') or not os.path.exists('./goods_encoder.pkl'):
        logger.info('---------------->>     create encoder    <<----------------')
        if len(user_ls) == 0:
            vdp = VIPDataProcessor(config, mode='train')
            _, user_ls, goods_ls = vdp.fe_process(save=True)
        test_data = pd.read_csv(os.path.join(processed_train_path, 'data.csv')).reset_index(drop=True)
        user_ls = user_ls + list(test_data['user_id'].unique())
        goods_ls = goods_ls + list(test_data['goods_id'].unique())
        u_encoder, g_encoder = encode(user_ls, goods_ls)

        logger.info('---------------->>      encode data      <<----------------')
        train_data['user_id'] = u_encoder.transform(train_data['user_id'])
        train_data['goods_id'] = g_encoder.transform(train_data['goods_id'])
        with open('./user_encoder.pkl', 'wb') as f:
            pickle.dump(u_encoder, f)
            f.close()
        with open('./goods_encoder.pkl', 'wb') as f:
            pickle.dump(g_encoder, f)
            f.close()
    else:
        logger.info('---------------->>      encode data      <<----------------')
        with open('./user_encoder.pkl', 'rb') as f:
            u_encoder = pickle.load(f)
            f.close()
        with open('./goods_encoder.pkl', 'rb') as f:
            g_encoder = pickle.load(f)
            f.close()
        train_data['user_id'] = u_encoder.transform(train_data['user_id'])
        train_data['goods_id'] = g_encoder.transform(train_data['goods_id'])


    paddle.seed(seed)

    model_class = DynamicModel(config)

    model = model_class.create_model()
    optimizer = model_class.create_optimizer(model)

    neg = train_data[train_data['is_order_max'] == 0].sample(frac=sampling_rate)
    pos = train_data[train_data['is_order_max'] != 0]

    train_data = pd.concat([pos, neg]).sample(frac=1).reset_index(drop=True)

    for epoch_id in range(num_epochs):
        model.train()
        metric_list, metric_list_name = model_class.create_metrics()

        epoch_begin = time.time()
        interval_begin = time.time()
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        reader_start = time.time()

        step_num = 0
        loader = get_batch_data(config, train_data, mode='train')
        for batch_id, batch in enumerate(loader):
            train_reader_cost += time.time() - reader_start
            optimizer.clear_grad()
            train_start = time.time()

            batch_size = len(batch[0])

            loss, metric_list, tensor_print_dict = model_class.train_forward(model, metric_list, batch)

            loss.backward()
            optimizer.step()

            train_run_cost += time.time() - train_start
            total_samples += batch_size

            if batch_id % print_interval == 0:
                metric_str = ""

                f1 = f1_score(metric_list[0].accumulate(), metric_list[1].accumulate())
                metric_str += ('F1' + ":{:.6f}, ".format(f1))

                tensor_print_str = ""
                if tensor_print_dict is not None:
                    for var_name, var in tensor_print_dict.items():
                        tensor_print_str += ("{}:".format(var_name) + str(var.numpy()).strip("[]") + ",")

                logger.info(
                    "epoch: {}, batch_id: {}, ".format(epoch_id, batch_id) + metric_str + tensor_print_str +
                    " avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.5f} ins/s".
                    format(
                        train_reader_cost / print_interval,
                        (train_reader_cost + train_run_cost) / print_interval,
                        total_samples / print_interval,
                        total_samples / (train_reader_cost + train_run_cost + 0.0001)
                    )
                )

                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0

            reader_start = time.time()
            step_num += 1

        metric_str = ""
        for metric_id in range(len(metric_list_name)):
            metric_str += (metric_list_name[metric_id] + ": {:.6f},".format(metric_list[metric_id].accumulate()))
            metric_list[metric_id].reset()

        tensor_print_str = ""
        if tensor_print_dict is not None:
            for var_name, var in tensor_print_dict.items():
                tensor_print_str += ("{}:".format(var_name) + str(var.numpy()).strip("[]") + ",")

        logger.info(
            "epoch: {} done, ".format(epoch_id) + metric_str + tensor_print_str + " epoch time: {:.2f} s".format(
                time.time() - epoch_begin)
        )

        save_model(model, optimizer, model_save_path, epoch_id, prefix='rec')
