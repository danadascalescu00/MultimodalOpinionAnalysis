from helpers import *

class Parameters():
    def __init__(self):
        self.SEED = 42
        self.use_cuda = False
        self.batch_size = 32

        self.cwd = os.path.dirname(os.path.abspath(__file__))
        self.dir_save_files = os.path.join(self.cwd, 'SaveFiles')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {}'.format(self.dir_save_files))
        else:
            print('directory {} exists'.format(self.dir_save_files))

        self.epochs = 10
        self.print_every = 15
        self.num_epochs_freeze_bert = 1