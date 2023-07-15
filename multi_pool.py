from multiprocessing import Pool, cpu_count

import pandas as pd


def run_multi_pool(function, file_name, processor=cpu_count() - 2):
    res = []
    p = Pool(processor)

    for i in range(processor):
        res.append(p.apply_async(function, args=(i,)))
        print(str(i) + ' processor started !')

    p.close()
    p.join()

    data = pd.concat([i.get() for i in res])
    data.to_csv(f'./训练集/{file_name}.csv', index=False)
