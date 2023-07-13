import paddle


class Tritower(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        raise NotImplementedError

    def forward(self, user, good):
        raise NotImplementedError
