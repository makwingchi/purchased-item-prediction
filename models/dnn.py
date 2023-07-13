import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class DNN(nn.Layer):
    def __init__(self, config):
        super().__init__()

        self.num_users = config['model']['baseline']['num_users']
        self.num_goods = config['model']['baseline']['num_goods']
        self.embedding_size = config['model']['baseline']['sparse_feature_dim']
        self.fc = config['model']['baseline']['fc_size']
        self.act = config['model']['baseline']['activate']

        self.user_embedding = nn.Embedding(
            self.num_users,
            self.embedding_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()
            )
        )

        self.goods_embedding = nn.Embedding(
            self.num_goods,
            self.embedding_size,
            weight_attr=paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierUniform()
            )
        )

        hidden_layers = [2 * self.embedding_size] + self.fc + [1]

        self.layers = []
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i],
                                         hidden_layers[i + 1],
                                         weight_attr=paddle.framework.ParamAttr(
                                             initializer=paddle.nn.initializer.XavierUniform()),
                                         bias_attr=paddle.ParamAttr(
                                             initializer=paddle.nn.initializer.Constant(value=0.0))
                                         ))

        # self.layers = nn.Sequential(*layers)

    def forward(self, data):
        user, goods = data[:, 0], data[:, 1]
        # feat = paddle.stack(feat, 1).astype(paddle.float32)

        user_vector = self.user_embedding(user)
        goods_vector = self.goods_embedding(goods)
        x = paddle.concat([user_vector, goods_vector], 1)

        for idx, layer in enumerate(self.layers):
            x = layer(x)

            if self.act == 'relu' and idx != len(self.layers) - 1:
                x = F.relu(x)

        return F.sigmoid(x)