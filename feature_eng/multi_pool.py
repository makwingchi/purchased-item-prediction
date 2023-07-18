import os
import datetime

from multiprocessing import cpu_count, Pool

import pandas as pd


def run_multi_pool(function, file_name, processor=range(22)):
    if not os.path.exists(f'./训练集/{file_name}'):
        os.mkdir(f'./训练集/{file_name}')

    p = Pool(len(processor))

    p.map(function, [i for i in processor])

    p.close()
    p.join()
