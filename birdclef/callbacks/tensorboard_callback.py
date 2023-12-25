from torch.utils.tensorboard import SummaryWriter

from birdclef.callbacks import callback


class TensorBoardCallback:

    def __init__(self, output_dir=''):
        super(TensorBoardCallback, self).__init__()
        self.writer = SummaryWriter(output_dir)

    def on_epoch_begin(self, epoch, **kwargs):
        pass

    def on_epoch_end(self, epoch, train_history, **kwargs):
        val_history = kwargs.get('val_history', None)
        for key, value in train_history.items():
            self.writer.add_scalar(str(key) + '/' + 'train', value, epoch)
        if val_history is not None:
            for key, value in val_history.items():
                self.writer.add_scalar(str(key) + '/' + 'val', value, epoch)
        self.writer.flush()

    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        self.writer.close()
