import paddle

from .bitower import Bitower
from .dnn import DNN
from .tritower import Tritower


class RecModel:
    def __init__(self, config):
        self.config = config
        self.model_type = config["runner"]["model_type"]

    def __get_model(self):
        _map = {
            "baseline": DNN,
            "bitower": Bitower,
            "tritower": Tritower,
        }

        return _map[self.model_type]

    def create_model(self):
        model = self.__get_model()
        return model(self.config)

    def create_loss(self, pred, label):
        cost = paddle.nn.functional.log_loss(
            input=pred,
            label=paddle.cast(label, dtype="float32")
        )

        return paddle.mean(x=cost)

    def create_optimizer(self, model):
        learning_rate = self.config["optimizer"]["learning_rate"]
        optimizer = paddle.optimizer.Adam(
            learning_rate=learning_rate,
            parameters=model.parameters()
        )

        return optimizer

    def create_metrics(self):
        metric_list_name = ["F1"]
        precision = paddle.metric.Precision()
        recall = paddle.metric.Recall()
        return [precision, recall], metric_list_name
