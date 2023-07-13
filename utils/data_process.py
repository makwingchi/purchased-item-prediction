import datetime as dtt
import logging
import os

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt='[%Y-%m-%d %H:%M:%S]',
)
logger = logging.getLogger(__name__)


class VIPDataProcessor:
    def __init__(self, config, mode='train'):
        if mode == 'train':
            # training data
            train_g_path = config['runner']['train_goods_path']
            tmp_ls = os.listdir(train_g_path)
            file_ls = []
            logger.info('>> read goods data')
            for _f in tqdm(tmp_ls):
                if 'part' in _f:
                    file_ls.append(pd.read_csv(os.path.join(train_g_path, _f),
                                               header=None,
                                               names=['goods_id', 'cat_id', 'brandsn']
                                               ))
            self.train_goods = pd.concat(file_ls, axis=0)
            del file_ls

            train_u_path = config['runner']['train_users_path']
            tmp_ls = os.listdir(train_u_path)
            file_ls = []
            logger.info('>> read user data')
            for _f in tqdm(tmp_ls):
                if 'part' in _f:
                    file_ls.append(pd.read_csv(os.path.join(train_u_path, _f),
                                               header=None,
                                               names=['user_id', 'goods_id', 'is_clk',
                                                      'is_like', 'is_addcart', 'is_order',
                                                      'expose_start_time', 'dt'],
                                               nrows=None))

            self.train_user = pd.concat(file_ls, axis=0)
            del file_ls

            self.processed_train_save_path = config['runner']['processed_train']

        else:
            train_u_path = config['runner']['train_users_path']
            tmp_ls = os.listdir(train_u_path)
            file_ls = []
            logger.info('>> read goods data')
            for _f in tqdm(tmp_ls):
                if 'part' in _f:
                    file_ls.append(pd.read_csv(os.path.join(train_u_path, _f),
                                               header=None,
                                               names=['user_id', 'goods_id', 'is_clk',
                                                      'is_like', 'is_addcart', 'is_order',
                                                      'expose_start_time', 'dt'],
                                               nrows=None))

            self.train_user = pd.concat(file_ls, axis=0)
            del file_ls
            self.train_user['expose_start_time'] = pd.to_datetime(self.train_user['expose_start_time'])
            self.candidate = self.train_user[~(
                    (self.train_user['expose_start_time'] >= self.train_user['expose_start_time'].max() + dtt.timedelta(days=-3)
                     ) & (self.train_user['is_order'] == 1))][['user_id', 'goods_id']].drop_duplicates()
            self.processed_test_save_path = config['runner']['processed_test']

            # testing data
            # test_g_path = config['runner']['test_goods_path']
            # file_ls = os.listdir(test_g_path)
            # self.test_goods = pd.concat([
            #     pd.read_csv(os.path.join(test_g_path, _f), header=None, names=['goods_id', 'cat_id', 'brandsn']) for _f in file_ls
            # ], axis=0)
            #
            # test_u_path = config['runner']['test_user_path']
            # if os.listdir(test_u_path):
            #     file_ls = os.listdir(test_u_path)
            #     self.test_users = pd.concat([pd.read_excel(os.path.join(test_u_path, _f)) for _f in file_ls])
            # else:
            #     self.test_users = pd.read_excel(test_u_path)

    def get_test_candidate(self, save=False):
        if save:
            if not os.path.exists(self.processed_test_save_path):
                os.makedirs(self.processed_test_save_path)
            self.candidate.to_csv(os.path.join(self.processed_test_save_path, 'data.csv'), index=False)
            logging.info('>> save complete')
        return self.candidate

    def fe_process(self, save=False):
        """
        feature engineering

        ① user field
            1. statistical
                 1,3,7,15,28 bin(cnt/max/mean/min) -- clk/like/addcart/order
                 1,3,7,15,28 bin(cum/day has action) -- clk/like/addcart/order
            2. agg by
                category/brand
                1,3,7,15,28 bin(cnt/max/mean/min)
                1,3,7,15,28 (most clk/like/addcart/order)

        ② item field
            1.




        :param save:
        :return:
        """

        train_data = pd.merge(self.train_user.iloc[:], self.train_goods.iloc[:], on='goods_id')
        logging.info('>> merge complete')
        train_agg_feat = train_data.iloc[:].groupby(['user_id', 'goods_id']).agg({
            'is_clk': ['sum', 'max'],
            'is_like': ['sum', 'max'],
            'is_addcart': ['sum', 'max'],
            'is_order': ['sum', 'max'],
        })
        train_agg_feat = train_agg_feat.reset_index()
        train_agg_feat.columns = [
            'user_id',
            'goods_id',
            'is_clk_sum',
            'is_clk_max',
            'is_like_sum',
            'is_like_max',
            'is_addcart_sum',
            'is_addcart_max',
            'is_order_sum',
            'is_order_max'
        ]
        # test_goods_id_agg = train_agg_feat.groupby('goods_id').agg({
        #     'is_clk_sum': 'sum',
        #     'is_order_sum': 'sum'
        # })
        logging.info('>> agg complete')
        # test_goods_id_agg = test_goods_id_agg[test_goods_id_agg['is_clk_sum'] > 100]
        # test_goods_id_agg = test_goods_id_agg[test_goods_id_agg['is_order_sum'] > 0]

        if save:
            if not os.path.exists(self.processed_train_save_path):
                os.makedirs(self.processed_train_save_path)
            train_agg_feat.to_csv(os.path.join(self.processed_train_save_path, 'data.csv'), index=False)
            logging.info('>> save complete')
        return train_agg_feat, \
               list(self.train_user['user_id']),\
               list(self.train_user['goods_id']) + list(self.train_goods['goods_id'])
