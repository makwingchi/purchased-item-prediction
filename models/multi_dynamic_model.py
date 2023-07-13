import paddle
import logging
from .rec_model import RecModel
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt='[%Y-%m-%d %H:%M:%S]',
)
logger = logging.getLogger(__name__)


class MultiDynamicModel(RecModel):
    def __init__(self, config):
        super().__init__(config)

    def create_feeds(self, batch_data):
        label = batch_data[:4]
        features = batch_data[4:]
        return label, features

    def train_forward(self, model, metric_list, batch_data):
        model.train()

        label, features = self.create_feeds(batch_data)
        pred = model.forward(features[0])
        loss = self.create_loss(pred, label)

        # predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metric_list[0].update(preds=pred.numpy(), labels=label.numpy())
        metric_list[1].update(preds=pred.numpy(), labels=label.numpy())

        print_dict = {'loss': loss}

        return loss, metric_list, print_dict

    def infer_forward(self, model, metric_list, batch_data):
        model.eval()

        label, features = self.create_feeds(batch_data)
        pred = model.forward(features)

        # predict_2d = paddle.concat(x=[1 - pred, pred], axis=1)
        metric_list[0].update(preds=pred.numpy(), labels=label.numpy())
        metric_list[1].update(preds=pred.numpy(), labels=label.numpy())

        return metric_list, None, pred.reshape(shape=[-1, ]).tolist()
