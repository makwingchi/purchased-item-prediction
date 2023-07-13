import os

import pandas as pd
from config import get_configurations

THRESHOLD = 0.5

if __name__ == '__main__':
    config = get_configurations()
    processed_test_path = config['runner']['processed_test']
    test_data = pd.read_csv(os.path.join(processed_test_path, 'data.csv'),
                            usecols=['user_id', 'goods_id']).reset_index(drop=True)
    infer_df = pd.read_csv('./infer.csv')
    test_data['pred'] = infer_df['pred']

    re_data = test_data[test_data['pred'] >= THRESHOLD]
    re_data.to_csv('./u2i.csv', header=False)
