import abc


class Callback(metaclass=abc.ABCMeta):

    def __init__(self):
        pass

    @abc.abstractmethod
    def on_train_begin(self, **kwargs):
        pass

    @abc.abstractmethod
    def on_train_end(self, **kwargs):
        pass

    @abc.abstractmethod
    def on_epoch_begin(self, **kwargs):
        pass

    @abc.abstractmethod
    def on_epoch_end(self, **kwargs):
        pass
