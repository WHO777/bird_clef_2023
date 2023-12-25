import csv
from pathlib import Path

from birdclef.callbacks import callback


class CSVLoggerCallback(callback.Callback):

    def __init__(self, output_dir='', filename='history.csv'):
        super(CSVLoggerCallback, self).__init__()
        self.filepath = str(Path(output_dir) / filename)

    def on_train_begin(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_epoch_begin(self, epoch, **kwargs):
        pass

    def on_epoch_end(self, epoch, train_history, **kwargs):
        val_history = kwargs.get('val_history', None)
        with open(self.filepath, 'a') as f:
            writer = csv.writer(f, delimiter=',')
            if epoch == 0:
                fieldnames = ['epoch'] + [
                    'train_' + key if key != 'learning_rate' else key
                    for key in train_history.keys()
                ]
                if val_history is not None:
                    fieldnames += ['val_' + key for key in val_history.keys()]
                writer.writerow(fieldnames)
            row = [epoch] + list(train_history.values())
            if val_history is not None:
                row += val_history.values()
            writer.writerow(row)
