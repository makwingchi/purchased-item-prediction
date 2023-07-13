import os
import pickle
import time
import logging
# import argparse
import datetime as dtt

from utils.data_process import VIPDataProcessor
from utils.tools import f1_score

dtt.timedelta
import pandas as pd

import paddle

from utils.data_loader import RecDataset, get_batch_data
from utils.save_and_load import load_model
from config import get_configurations
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
    print_interval = config["runner"]["print_interval"]
    start_epoch = config["runner"]["infer_start_epoch"]
    end_epoch = config["runner"]["infer_end_epoch"]
    model_load_path = config["runner"]["model_save_path"]
    infer_batch_size = config["runner"]["batch_size"]
    # task_type = config["runner"]["task_type"]
    processed_test_path = config['runner']['processed_test']

    if not os.path.exists(processed_test_path):
        logger.info('---------------->> genereate testing set <<----------------')
        vdp = VIPDataProcessor(config, mode='test')
        test_data = vdp.get_test_candidate(save=True)
        logger.info('---------------->>     genereate end     <<----------------')
        del vdp
    else:
        logger.info('---------------->>   load testing set    <<----------------')
        test_data = pd.read_csv(os.path.join(processed_test_path, 'data.csv')).reset_index(drop=True)

    logger.info('---------------->>      encode data      <<----------------')
    with open('./user_encoder.pkl', 'rb') as f:
        u_encoder = pickle.load(f)
        f.close()
    with open('./goods_encoder.pkl', 'rb') as f:
        g_encoder = pickle.load(f)
        f.close()

    test_data['user_id'] = u_encoder.transform(test_data['user_id'])
    test_data['goods_id'] = g_encoder.transform(test_data['goods_id'])
    del u_encoder, g_encoder

    paddle.seed(seed)

    model_class = DynamicModel(config)

    model = model_class.create_model()

    loader = get_batch_data(config, test_data, mode='infer')

    epoch_begin = time.time()
    interval_begin = time.time()

    metric_list, metric_list_name = model_class.create_metrics()
    step_num = 0

    for epoch_id in range(start_epoch, end_epoch):
        logger.info("load model epoch {}".format(epoch_id))
        model_path = os.path.join(model_load_path, str(epoch_id))
        load_model(model_path, model)

        model.eval()
        infer_reader_cost = 0.0
        infer_run_cost = 0.0
        reader_start = time.time()

        pred_ls = []

        for batch_id, batch in enumerate(loader):
            infer_reader_cost += time.time() - reader_start
            infer_start = time.time()
            batch_size = len(batch[0])

            metric_list, tensor_print_dict, curr_pred = model_class.infer_forward(model, metric_list, batch)
            pred_ls.extend(curr_pred)

            infer_run_cost += time.time() - infer_start

            if batch_id % print_interval == 0:
                tensor_print_str = ""
                if tensor_print_dict is not None:
                    for var_name, var in tensor_print_dict.items():
                        tensor_print_str += ("{}:".format(var_name) + str(var.numpy()).strip("[]") + ",")

                metric_str = ""
                for metric_id in range(len(metric_list_name)):
                    metric_str += (metric_list_name[metric_id] + ": {:.6f},".format(metric_list[metric_id].accumulate()))

                logger.info(
                    "epoch: {}, batch_id: {}, ".format(epoch_id, batch_id) + metric_str + tensor_print_str +
                    " avg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {:.5f}, ips: {:.2f} ins/s".
                    format(
                        infer_reader_cost / print_interval,
                        (infer_reader_cost + infer_run_cost) / print_interval,
                        infer_batch_size,
                        print_interval * batch_size / (time.time() + 0.0001 - interval_begin)
                    )
                )
                interval_begin = time.time()
                infer_reader_cost = 0.0
                infer_run_cost = 0.0
            step_num = step_num + 1
            reader_start = time.time()

        metric_str = ""

        f1 = f1_score(metric_list[0].accumulate(), metric_list[1].accumulate())
        metric_str += ('F1' + ":{:.6f}, ".format(f1))
        metric_list[0].reset()
        metric_list[1].reset()

        tensor_print_str = ""
        if tensor_print_dict is not None:
            for var_name, var in tensor_print_dict.items():
                tensor_print_str += ("{}:".format(var_name) + str(var.numpy()).strip("[]") + ",")

        logger.info(
            "epoch: {} done, ".format(epoch_id) + metric_str + tensor_print_str + " epoch time: {:.2f} s".format(time.time() - epoch_begin)
        )
        epoch_begin = time.time()

        pd.DataFrame({"pred": pred_ls}).to_csv("./infer.csv", index=False)
