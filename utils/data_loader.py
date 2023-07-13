import logging
import os
import pickle
import numpy as np
import paddle
import pandas as pd
from paddle.io import IterableDataset, Dataset
from utils.tools import encode

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# class RecDataset(IterableDataset):
#     def __init__(self, config, data, mode):
#         super().__init__()
#         self.max_len = config["runner"]["max_len"]
#         self.seed = config["runner"]["seed"]
#         self.sampling_rate = config["runner"]["sampling_rate"]
#         self.data = data
#         self.mode = mode

#     def __iter__(self):
#         for _, row in self.data.iterrows():
#             # logger.info(row.values[-1])
#             if self.mode == 'train':
#                 if row.values[-1] == 0 and np.random.random() > self.sampling_rate:
#                     continue
#                 yield [row.values[-1].astype("float32").reshape([-1,]), row.values[:-1]]
#             else:
#                 yield [np.array([0]).astype("float32").reshape([-1,]), row.values]

#     def sampling(self):
#         # TODO down-sampling
#         raise NotImplementedError

#     def __getitem__(self, index):
#         return self.data.loc[index]

#     def __len__(self):
#         return len(self.data)

# def get_encoder(self):
#     return self.u_encoder, self.g_encoder

class RecDataset(Dataset):
    def __init__(self, config, data, mode):
        self.data = data
        self.max_len = config["runner"]["max_len"]
        self.seed = config["runner"]["seed"]
        self.sampling_rate = config["runner"]["sampling_rate"]
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == 'train':
            feat = self.data.loc[idx].values[:-1].astype("int32").reshape([-1, ])
            label = np.array([self.data.loc[idx, 'is_order_max']]).astype("float32").reshape([-1, ])

            return [label, feat]
        else:
            # logger.info(self.data.loc[idx].values[:2])
            feat = self.data.loc[idx].values[:2].astype("int32").reshape([-1, ])
            label = np.array([0]).astype("float32").reshape([-1, ])
            return [label, feat]

    def __len__(self):
        return self.data.shape[0]


def get_batch_data(config, train_data, mode='train'):
    recdataset = RecDataset(config, train_data, mode)
    loader = paddle.io.DataLoader(
        recdataset,
        batch_size=config['runner']['batch_size'],
        places=config['runner']['device'],
        shuffle=False,
        drop_last=False
    )

    return loader