import csv
import os
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, logdir):
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)
        self.data = defaultdict(list)
        self.filepath = os.path.join(logdir, 'progress.csv')
        self.header_written = False
        self.tb_writer = SummaryWriter(log_dir=logdir)

    def log(self, key, value):
        self.data[key].append(value)

    def write(self, step=None):
        if not self.header_written:
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.data.keys())
                writer.writeheader()
            self.header_written = True

        row = {k: v[-1] for k, v in self.data.items()}
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.data.keys())
            writer.writerow(row)

        for k, v in row.items():
            self.tb_writer.add_scalar(k, v, step if step is not None else len(self.data[k]))

        if step is not None:
            print(f'Step {step}:', end=' ')
        else:
            print('Logging:', end=' ')
        print(' | '.join([f'{k}: {v:.3f}' for k, v in row.items()]))
