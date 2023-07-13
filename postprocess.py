import pandas as pd

THRESHOLD = 0.5

if __name__ == "__main__":
    test = pd.read_csv('./process_test/data.csv')
    tmp = pd.read_csv('./infer.csv')
    test['pred'] = tmp['pred']
    test[test['pred'] > THRESHOLD][['user_id', 'goods_id']].to_csv('./u2i.csv', index=False)
