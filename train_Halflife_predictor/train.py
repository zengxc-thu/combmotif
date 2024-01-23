# -*- coding = utf-8 -*-
# @Time : 2023/1/22 18:37
# @Author : zxc
# @File : my_training.py
# @Software :

import h5py
import torch
import datetime
import os
from torchinfo import summary
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('/mnt/disk1/xzeng/postgraduate/saluki_torch')
import utils
from utils import get_dataloader,get_eval_dataloader,calculate_accuracy
from model_zoo import saluki_torch
import torch.nn as nn
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import scipy.stats as stats
import matplotlib
import seaborn as sns

def r2(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value ** 2
def r(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return r_value

def run_a_train_epoch(model, training_dataloader, optimizer, device, loss_func,scheduler=None,args=None):
    model.train()
    mean_loss = torch.zeros(1).to(device)
    for step, (batch_x, batch_y) in enumerate(training_dataloader):  
        model.zero_grad()
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        output = model(batch_x)
        loss = loss_func(output, batch_y)  #
        loss.backward()  # backpropagation, compute gradients
        mean_loss = (mean_loss * step+ loss.detach()) / (step + 1)
        optimizer.step()  # apply gradients
    if("lr_decay_schedule" in args and args.lr_decay_schedule != None):scheduler.step()
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


    # loading sequence
    file = h5py.File(args.data_path, 'r')
    train_sequence = file['train_sequence'] 
    train_sequence = torch.from_numpy(train_sequence[:]).float()

    valid_sequence = file['valid_sequence'] 
    valid_sequence = torch.from_numpy(valid_sequence[:]).float()

    test_sequence = file['test_sequence'] 
    test_sequence = torch.from_numpy(test_sequence[:]).float()

    # loading halflife
    train_label = file['train_label'] 
    train_label =  torch.from_numpy(train_label[:]).unsqueeze(1).float()
    valid_label = file['valid_label'] 
    valid_label = torch.from_numpy(valid_label[:]).unsqueeze(1).float()  
    test_label = file['test_label'] 
    test_label = torch.from_numpy(test_label[:]).unsqueeze(1).float()  


    logger.info('train_data_shape: {}'.format(train_sequence.shape))
    logger.info('valid_data_shape: {}'.format(valid_sequence.shape))
    logger.info('test_data_shape: {}'.format(test_sequence.shape))

    train_dataloader = get_dataloader(train_sequence, train_label, args.batch_size)

    assert(args.seq_depth == test_sequence.shape[1])
    assert(args.seq_length == test_sequence.shape[2])

    model = saluki_torch(num_layers=args.num_layer, 
                          padding=args.padding,
                          dropout=args.dropout,
                          num_targets=args.num_targets,
                          seq_depth=args.seq_depth,
                          seq_length=args.seq_length)
    
    if("pretrain_weights" in args and args.pretrain_weights != None):
        model.load_state_dict(torch.load(args.pretrain_weights, map_location='cpu')['model_state_dict'])

    model.to(device)

    logger.info('\n----------Model details---------\n')
    logger.info(model)
    my_summary = summary(model,input_data=torch.zeros([args.batch_size,valid_sequence.shape[1],valid_sequence.shape[2]]).to(torch.float32).to(device), depth=10)
    logger.info(my_summary)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info('total numebr of parameters: {}'.format(total_params))
    
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.l2  
                                 )  # optimize all cnn parameters

    loss_func = nn.MSELoss()

    filename = os.path.join(args.output_dir, '%s-best.pth' % (type(model).__name__) )

    stopper = utils.EarlyStopping(mode='lower', patience=args.patience, tolerance=0.0,
                            filename=filename,logger=logger)
    
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []

    for epoch in range(1, args.epochs + 1):
        # train
        loss_train = run_a_train_epoch(model, train_dataloader, optimizer, device, 
                                       loss_func=loss_func,args=args,scheduler=scheduler,
                                       )
        train_label_pred = run_a_eval_epoch(model, train_sequence, args.batch_size, device=device)
        valid_label_pred = run_a_eval_epoch(model, valid_sequence, args.batch_size, device=device)

        loss_valid = loss_func(valid_label_pred, valid_label.to(device))

        valid_label_pred = valid_label_pred.cpu()
        train_label_pred = train_label_pred.cpu()
        r2_train = r2(train_label.ravel(), train_label_pred.ravel())
        r_train = r(train_label.ravel(), train_label_pred.ravel())
        r2_valid = r2(valid_label.ravel(), valid_label_pred.ravel())
        r_valid = r(valid_label.ravel(), valid_label_pred.ravel())

        early_stop = stopper.step(loss_valid, model, logger)
        logger.info ('EPOCH%d train_loss = %.4f train_r2 = %.4f train_r = %.4f  valid_loss = %.4f  valid_r2 = %.4f valid_r = %.4f   ' % (epoch, loss_train, r2_train, r_train, loss_valid, r2_valid, r_valid) )
        if early_stop:
            break


    stopper.load_checkpoint(model)

    test_label_pred = run_a_eval_epoch(model, test_sequence, args.batch_size, device=device)
    test_label_pred = test_label_pred.cpu()
    r2_test = r2(test_label.ravel(), test_label_pred.ravel())
    r_test = r(test_label.ravel(), test_label_pred.ravel())

    print("test_r:%.4f" % r_test)
    print("test_r2:%.4f \t  " % r2_test)


    testdata = pd.DataFrame({'pred': test_label_pred.ravel(), 'label': test_label.ravel()})


    font = {'size' : 18}
    matplotlib.rc('font', **font)
    matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})   
    fig, ax = plt.subplots(figsize = (6,4), dpi = 300)

    c1 = "tab:blue" # (0.3, 0.45, 0.69)# cornflowerblue, royalblue
    # c1 = "tab:green"
    c2 = 'r'
    g = sns.JointGrid(x='label', y="pred", data=testdata, space=0, xlim=(1,10), ylim=(0,10), ratio=6, height=7)
    g.plot_joint(plt.scatter,s=20, color=c1, alpha=1, linewidth=0.4, edgecolor='black',) #

      
    f = g.fig
    ax = f.gca()
    ax.set_ylim(-4,4)
    ax.set_xlim(-4,4)
    ax.text(x=.71, y=0.03,s='r: ' + str(round(r_test, 2)), transform=ax.transAxes) # size
    g.plot_marginals(sns.kdeplot,fill=r_test, **{'linewidth':2, 'color':c1})
    g.set_axis_labels('Observed halflife', 'Predicted halflife', **{'size':18}) # **{'size':22}

    f = g.fig

    current_time = datetime.now()
    image_filename = current_time.strftime('%Y%m%d%H%M%S') + '_scatter.png'
    plt.savefig(os.path.join(args.output_dir, image_filename))


