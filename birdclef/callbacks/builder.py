from birdclef.callbacks.csv_logger import CSVLoggerCallback
from birdclef.callbacks.tensorboard_callback import TensorBoardCallback


def build_callbacks(configs):
    callbacks = []
    for config in configs:
        callback_type = eval(config.type)
        del config.type
        callback = callback_type(**config)
        callbacks.append(callback)
    return callbacks
