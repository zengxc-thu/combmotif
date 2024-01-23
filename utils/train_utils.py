import torch.utils.data as Data
import torch
import numpy as np
import random
import logging
def getLogger(work_dir,time_stamp=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    # StreamHandler
    sHandler = logging.StreamHandler()
    # sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)


    fHandler = logging.FileHandler(work_dir + '/%s.txt'%time_stamp, mode='w')
    fHandler.setLevel(logging.DEBUG) 

    logger.addHandler(fHandler) 

    return logger
def record_args(args, logger):
    logger.info('\n----------Input arguments---------\n')
    for key, value in args.items():
        logger.info('%s : %s' %(key, str(value)))

def get_dataloader(x, y, args_batch_size, shuffle=True):


    torch_dataset = Data.TensorDataset(x, y)

    loader = Data.DataLoader(
        dataset=torch_dataset,      # torch TensorDataset format
        batch_size=args_batch_size,      # mini batch size
        shuffle=shuffle,               # random shuffle for training
        num_workers=2,              # subprocesses for loading 100wdata
    )

    return loader

def get_eval_dataloader(x, args_batch_size, shuffle=False):


    torch_dataset = Data.TensorDataset(x)

    loader = Data.DataLoader(
        dataset=torch_dataset,      # torch TensorDataset format
        batch_size=args_batch_size,      # mini batch size
        shuffle=shuffle,               # random shuffle for training
        num_workers=0,              # subprocesses for loading 100wdata
    )

    return loader

class EarlyStopping(object):
    def __init__(self,  mode='higher', patience=15, filename=None, tolerance=0.0,logger=None):
        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.tolerance = tolerance
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        # return (score > prev_best_score)
        return score / prev_best_score > 1 + self.tolerance

    def _check_lower(self, score, prev_best_score):
        # return (score < prev_best_score)
        return prev_best_score / score > 1 + self.tolerance

    def step(self, score, model,logger):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, self.filename) 

    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])

def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
def calculate_accuracy(pred, true):
    pred = torch.round(pred)
    num = pred.shape[0]

    accuracy = 1- torch.sum(torch.abs(pred - true))/num
    return  accuracy

def weights_init(net, init_type='normal', init_gain=0.001):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, 0.001)
                torch.nn.init.normal_(m.bias.data, 0.1)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.01)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 1 / (m.in_features**0.5))
            torch.nn.init.constant_(m.bias.data, 1.0)
    net.apply(init_func)