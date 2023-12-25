from birdclef.metrics.accuracy import Accuracy
from birdclef.metrics.auc import AUC, AUC_TF
from birdclef.metrics.padded_cmap import PaddedCMap


def build_metrics(configs):
    metrics = []
    for config in configs:
        metric_type = eval(config.type)
        del config.type
        metric = metric_type(**config)
        metrics.append(metric)

    return metrics
