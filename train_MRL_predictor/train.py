import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from sklearn import preprocessing
import random
import utils
from tqdm import tqdm
from utils import get_dataloader,get_eval_dataloader
import os
import datetime
import torch
import model_zoo
import torch.nn as nn
from torchinfo import summary
import torch.optim.lr_scheduler as lr_scheduler
import re
np.random.seed(1337)


### Parameters for plotting model results ###
pd.set_option("display.max_colwidth", 100)
sns.set(style="ticks", color_codes=True)
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.labelpad'] = 5
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'

def get_train_dataloader(df: pd.DataFrame, noAUG = False):

    if(not noAUG):
        #optimus 5 prime split config
        df.sort_values('total_reads', inplace=True, ascending=False)
        df.reset_index(inplace=True, drop=True)
        df = df.iloc[:280000]
        # The training set has 260k UTRs and the test set has 20k UTRs.
        e_train = df.iloc[20000:].copy()  
        e_test = df.iloc[:20000].copy()
    else:
        e_train = df.iloc[int(len(df)/10):].copy()  
        e_test = df.iloc[:int(len(df)/10)].copy()


    # One-hot encode both training and test UTRs
    seq_e_train = torch.from_numpy(one_hot_encode(e_train,seq_len=50)).permute(0,2,1).float()
    seq_e_test = torch.from_numpy(one_hot_encode(e_test,seq_len=50)).permute(0,2,1).float()


    # Scale the training mean ribosome load values
    
    e_train['scaled_rl'] = preprocessing.StandardScaler().fit_transform(e_train.copy().loc[:,'rl'].values.reshape(-1,1))

    logger.info('train_data_shape: {}'.format(seq_e_train.shape))
    train_dataloader = get_dataloader(seq_e_train, torch.from_numpy(e_train['scaled_rl'].values).float(), args.batch_size)

    test_scaler = preprocessing.StandardScaler()
    test_scaler.fit(e_test['rl'].values.reshape(-1, 1))

    test_label = e_test['rl']


    return train_dataloader,seq_e_test,test_scaler,test_label,

def run_a_train_epoch(model, training_dataloader, optimizer, device, loss_func,scheduler=None,args=None):
    model.train()
    mean_loss = torch.zeros(1).to(device)
    for step, (batch_x, batch_y) in enumerate(training_dataloader):  
        model.zero_grad()
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        output = model(batch_x)
        loss = loss_func(output.ravel(), batch_y)  #
        loss.backward()  # backpropagation, compute gradients
        mean_loss = (mean_loss * step+ loss.detach()) / (step + 1)
        optimizer.step()  # apply gradients
    if("lr_decay_schedule" in args and args.lr_decay_schedule):scheduler.step()
    return mean_loss
def run_a_eval_epoch(model, X, args_batch_size, device):
    dataloader = get_eval_dataloader(X, args_batch_size)
    model.eval()
    with torch.no_grad():
        pred = torch.tensor([])
        pred = pred.to(device)
        for step, (batch_x) in enumerate(dataloader):  
            batch_x = batch_x[0].to(device)
            output = model(batch_x)
            pred = torch.cat((pred, output), dim=0)
    return pred


def test_data(df, model, test_seq, obs_col, output_col='pred'):
    '''Predict mean ribosome load using model and test set UTRs'''

    # Scale the test set mean ribosome load
    scaler = preprocessing.StandardScaler()
    scaler.fit(df[obs_col].values.reshape(-1, 1))

    # Make predictions
    predictions = model.predict(test_seq).reshape(-1)

    # Inverse scaled predicted mean ribosome load and return in a column labeled 'pred'
    df.loc[:, output_col] = scaler.inverse_transform(predictions)
    return df


def one_hot_encode(df, col='utr', seq_len=50):
    # Dictionary returning one-hot encoding of nucleotides.
    nuc_d = {'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1], 'n': [0, 0, 0, 0]}

    # Creat empty matrix.
    vectors = np.empty([len(df), seq_len, 4])

    # Iterate through UTRs and one-hot encode
    for i, seq in enumerate(df[col].str[:seq_len]):
        seq = seq.lower()
        a = np.array([nuc_d[x] for x in seq])
        vectors[i] = a
    return vectors


def r2(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value ** 2


if __name__ == '__main__':
    
    current_folder = os.path.dirname(os.path.abspath(__file__))

    
    os.chdir(current_folder)

    args = utils.getargs()

    cfgs = utils.read_yaml_to_dict(args.config)
    
    for key in cfgs.keys():
        args[key] = cfgs[key]


    dt = datetime.datetime.now()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    time_stamp = '{}_{:02d}_{:02d}_{:02d}'.format(dt.date(), dt.hour, dt.minute, dt.second)

    args['dataset_name'] = args.data_path.split('/')[-1].split('.')[0]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    utils.save_dict_to_yaml(args,'%s/config.yaml'%args.output_dir)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # create logger
    logger = utils.getLogger(args.output_dir, time_stamp)
    logger.info('\n----------introduction---------\n')
    logger.info(args.description)
    utils.record_args(args, logger)
        
    ### Load data, make a train and test set based on total reads per UTR
    # The test set contains UTRs with the highest overall sequencing reads with the idea that increased reads will more accurately reflect the true ribosome load of a given 5'UTR.
    df = pd.read_csv(args.data_path)
    train_dataloader, test_sequence, test_scaler, test_label  = get_train_dataloader(df, args.noAUG)
    logger.info('test_data_shape: {}'.format(test_sequence.shape))
    
    ### Train model
    # Using the hyperparameter-optimised values.
    assert(args.seq_depth == test_sequence.shape[1])
    assert(args.seq_length == test_sequence.shape[2])

    model = model_zoo.saluki_torch(num_layers=args.num_layer, 
                          padding=args.padding,
                          dropout=args.dropout,
                          num_targets=args.num_targets,
                          seq_depth=args.seq_depth,
                          seq_length=args.seq_length)

    if("pretrain_weights" in args and args.pretrain_weights!=None):
        model.load_state_dict(torch.load(args.pretrain_weights, map_location='cpu')['model_state_dict'])

    model.to(device)

    logger.info('\n----------Model details---------\n')
    logger.info(model)
    my_summary = summary(model,input_data=torch.zeros([args.batch_size,4,50]).to(torch.float32).to(device), depth=10)
    logger.info(my_summary)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info('Total parameters：{}'.format(total_params))


    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.l2  
                                )  # optimize all cnn parameters
    loss_func = nn.MSELoss()

    filename = args.output_dir + '/' + '%s-%s-best.pth' % (type(model).__name__,args.dataset_name)
    stopper = utils.EarlyStopping(mode='lower', patience=args.patience, tolerance=0.0,
                            filename=filename,logger=logger)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in tqdm(range(0, args.epochs)):
        # train
        loss_train = run_a_train_epoch(model, train_dataloader, optimizer, device, loss_func=loss_func,args=args,scheduler=scheduler)
        test_label_pred_scalered = run_a_eval_epoch(model, test_sequence, args.batch_size, device=device)
        test_label_pred = test_scaler.inverse_transform(test_label_pred_scalered.cpu())
        r = r2(test_label, test_label_pred.ravel())
        early_stop = stopper.step(loss_train, model,logger)
        logger.info ('EPOCH%d train_loss = %.4f  test_r2 = %.4f' % (epoch, loss_train , r) )
        if early_stop:
            break
    

    stopper.load_checkpoint(model)
    
    
    test_label_pred_scalered = run_a_eval_epoch(model, test_sequence, args.batch_size, device=device)
    test_label_pred = test_scaler.inverse_transform(test_label_pred_scalered.cpu())


    test_label_pred = test_label_pred.ravel()

    r = r2(test_label, test_label_pred.ravel())
    logger.info('test r-squared = %.4f' % r)
    e_test = pd.DataFrame({'rl': test_label, 'pred': test_label_pred})



    import matplotlib
    # sns.set_style("darkgrid")
    font = {'size' : 18}
    matplotlib.rc('font', **font)
    matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
    fig, ax = plt.subplots(figsize = (6,4), dpi = 300)

    c1 = "tab:blue" # (0.3, 0.45, 0.69)# cornflowerblue, royalblue
    c2 = 'r'
    g = sns.JointGrid(x='rl', y="pred", data=e_test, space=0, xlim=(-10,10), ylim=(-10,10), ratio=6, height=7)
    g.plot_joint(plt.scatter,s=20, color=c1, linewidth=0.4, alpha=1, edgecolor='black')
    f = g.fig
    ax = f.gca()
    ax.set_yticks(np.arange(0,9.01, 1))
    ax.set_yticklabels(range(10),size=20)
    ax.set_xticks(np.arange(1,10.01, 1))
    ax.set_xticklabels(range(1,11),size=20)
    ax.set_ylim(0,9)
    ax.set_xlim(1,10)
    ax.text(x=.71, y=0.03,s='r$^2$: ' + str(round(r, 3)), transform=ax.transAxes)
    g.plot_marginals(sns.kdeplot,fill=r, **{'linewidth':2, 'color':c1})
    g.set_axis_labels('Observed MRL', 'Predicted MRL', **{'size':18})

    f = g.fig
    plt.savefig(args.output_dir + '/scatter.png')