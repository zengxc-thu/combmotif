# -*- coding = utf-8 -*-
# @Time : 2023/9/19 20:23
# @Author : zxc
# @File : search_maxact_trainset.py
# @Software :Find the maximum activation values for each neuron along with the corresponding activation samples and their corresponding labels from the training set.
from __future__ import division, print_function
import datetime
import joblib
import os
import utils
import re
import torch
from tqdm import tqdm
import h5py
from utils import resample_from_trainingset,get_activation
import numpy as np
from model_zoo import Saluki_Motif
import random
import sys
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

if __name__=='__main__':

 
    current_folder = os.path.dirname(os.path.abspath(__file__))

    
    os.chdir(current_folder)

    args = utils.get_explain_args()

    cfgs = utils.read_yaml_to_dict(args.config)
    for key in cfgs.keys():
        args[key] = cfgs[key]

    set_random_seed(args.seed)
    conv_layer_name = args.name

    print(f'Start searching the maxactivations for each neuron in {conv_layer_name} from training set...')

    sample_save_path = os.path.join(args.maxactivation, args.model_name)

    if not os.path.exists(sample_save_path):
        os.makedirs(sample_save_path)

    if os.path.exists(sample_save_path + '/' + conv_layer_name + '.pkl'):
        print(f'Maxactivations of the neurons in {conv_layer_name} exist, skip')
        sys.exit()

    dt = datetime.datetime.now()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    recp_field = utils.read_yaml_to_dict(args.recp_path)

    
    network = Saluki_Motif(num_layers=args.num_layer,
                                   seq_depth = args.seq_depth, 
                                   num_targets = args.num_targets)
    network.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model_state_dict'])
    network.to(device)
    network.eval()


    ##加载数据
    file = h5py.File(args.data_path, 'r')
    train_sequence = file['train_sequence'] 
    train_sequence = torch.from_numpy(train_sequence[:]).float()
    train_sequence = train_sequence[:,:4,:]
    train_label = file['train_label'] 
    train_label =  torch.from_numpy(train_label[:]).unsqueeze(1).float()


    iters = int(recp_field[conv_layer_name] / 10 * 400 / 5)
    n = 20000
    patience = args.search_patience
    cnt = 0
    for k in tqdm(range(iters)):
        xs = resample_from_trainingset(train_sequence, train_label, n, recp_field[conv_layer_name])
        pop = torch.from_numpy(xs).float()

        act = get_activation(pop.clone(),network,conv_layer_name, device)
        if k == 0:
            max_activation = torch.zeros([1,act.shape[-1]]).to(device)
            max_activation_samples = -torch.ones([act.shape[-1], train_sequence.shape[1], recp_field[conv_layer_name]])
            sample_labels = torch.zeros([act.shape[-1]])
            last_max_max_activation = max_activation.clone()

        all_act = torch.cat((act,max_activation),dim=0)
        all_activation_samples = torch.cat((pop, max_activation_samples),dim=0)
        index = all_act.max(dim=0).indices
        for i,ind in enumerate(index):
            # "i" represents the maximum activation value for the i-th neuron.
            # "ind" represents that the maximum value for the i-th neuron occurs in the output from the i-th sample.
            # If "ind" is equal to the last value in "all_act," which is the maximum value from the previous generation, it indicates that there is no need to change the values in "max_activation_samples."
            if(ind != all_act.shape[0] - 1):
                max_activation_samples[i] = all_activation_samples[ind]
  
                
                
        # sample_labels = all_labels[index]
        max_activation = torch.unsqueeze(all_act.max(dim=0).values,dim=0)
        
        # if (last_max_max_activation == max_activation).all():
        #     cnt += 1  
        #     print('count%d'%cnt)
        #     if cnt>patience:
        #         break
        # else:
        #     cnt=0
        
        last_max_max_activation = max_activation
        formatted_activations = [round(num, 2) for num in max_activation[0, :10].tolist()]
        print(f'iter{k}: the maxacts of {conv_layer_name}_neuron 1-10:{formatted_activations}')

        if(k%10==0 and k):
            joblib.dump(max_activation_samples, sample_save_path + '/' + conv_layer_name + '.pkl')
            joblib.dump(max_activation, sample_save_path + '/' + conv_layer_name + '_maxactivation.pkl')

    print('---------------%s MAX Activation-----------------'%conv_layer_name)
    print('%s'%(max_activation[0,:10]))

    print(f"saving maxactivations to {sample_save_path + '/' + conv_layer_name + '.pkl'}")

    joblib.dump(max_activation_samples, sample_save_path + '/' + conv_layer_name + '.pkl')







