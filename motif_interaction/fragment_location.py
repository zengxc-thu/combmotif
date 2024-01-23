




# -*- coding = utf-8 -*-
# @Time : 2023/9/25 20:23
# @Author : zxc
# @Software :For a specific neuron, if a segment of a sample in the training set can activate it to achieve more than half of its maximum activation value, then that sample is considered a positive sample. Conversely, it is considered a negative sample if it doesn't meet this criterion. Check if there is a difference in Mean Relative Logit (MRL) between these two groups of samples.
from __future__ import division, print_function
import os
import joblib
import utils
from tqdm import tqdm
import datetime
from model_zoo import Saluki_Motif
import h5py
import numpy as np
import random
import torch
import sys
from utils import get_activation
def run_cmd(cmd_str='', echo_print=0):

    from subprocess import run
    if echo_print == 1:
        print('\nExecute cmd instruction="{}"'.format(cmd_str))
    run(cmd_str, shell=True)
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

def cal_fragment_location(network,conv_layer_name, neuron_index, maxactivation, seq, recep_field,threshold,device):

    assert(seq.shape[1] >= recep_field)

    samples = torch.zeros(seq.shape[1] - recep_field + 1, seq.shape[0], recep_field)
    for i in range(seq.shape[1] - recep_field + 1):
        samples[i] = seq[:,i:i+recep_field]


    activation_value = get_activation(samples,network,conv_layer_name,device)

    activation_value = activation_value[:, neuron_index]
    

    return torch.where(activation_value>threshold*maxactivation)[0]

def cal_fragment_location_faster(network,conv_layer_name, maxactivations, seq, recep_field,threshold,device):

    assert(seq.shape[1] >= recep_field)

    samples = torch.zeros(seq.shape[1] - recep_field + 1, seq.shape[0], recep_field)
    for i in range(seq.shape[1] - recep_field + 1):
        samples[i] = seq[:,i:i+recep_field]


    activation_value = get_activation(samples,network,conv_layer_name,device)

    locs = []
    for i in range(maxactivations.shape[0]):
        loc = torch.where(activation_value[:, i]>threshold*maxactivations[i])[0]
        locs.append(loc)
    return locs

if __name__ == "__main__":
    
    current_folder = os.path.dirname(os.path.abspath(__file__))

    
    os.chdir(current_folder)

    args = utils.get_explain_args()

    cfgs = utils.read_yaml_to_dict(args.config)
    for key in cfgs.keys():
        args[key] = cfgs[key]

    conv_layer_name = args.name

    print(f'Start fragment location for {conv_layer_name}...')


    set_random_seed(args.seed)
    dt = datetime.datetime.now()

    
    recp_field = utils.read_yaml_to_dict(args.recp_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_network = Saluki_Motif(num_layers=args.num_layer,
                                   seq_depth = args.seq_depth, 
                                   num_targets = args.num_targets)
    sample_network.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model_state_dict'])
    sample_network.to(device)
    sample_network.eval()

    file = h5py.File(args.data_path, 'r')
    train_sequence = file['train_sequence'] 
    train_sequence = torch.from_numpy(train_sequence[:]).float()

    valid_sequence = file['valid_sequence'] 
    valid_sequence = torch.from_numpy(valid_sequence[:]).float()

    test_sequence = file['test_sequence'] 
    test_sequence = torch.from_numpy(test_sequence[:]).float()

    train_label = file['train_label'] 
    train_label =  torch.from_numpy(train_label[:]).float()
    valid_label = file['valid_label'] 
    valid_label = torch.from_numpy(valid_label[:]).float()  
    test_label = file['test_label'] 
    test_label = torch.from_numpy(test_label[:]).float()  

    seq = torch.cat((train_sequence, valid_sequence, test_sequence), dim=0)
    seq = seq[:,0:4,:] #Only take the first four dims because we are interested in the maximum activation value of neurons



    maxactivation_save_path = os.path.join(args.maxactivation, args.model_name, conv_layer_name + '_maxactivation.pkl')
    if(not os.path.exists(maxactivation_save_path)):
        print('*********No %s maxactivations found**********' %conv_layer_name)
        print('*********Plz run motif_interaction/script/run_search_maxact.sh first to obtain maxact**********' %conv_layer_name)
        sys.exit()

    maxactivations = joblib.load(maxactivation_save_path)

    maxactivations = maxactivations.ravel()

    neuron_num = maxactivations.shape[0]

    thresh_hold = 0.90

    save_dir = f'./{args.fragment}/{args.model_name}'
    if(not os.path.exists(save_dir)):
        os.makedirs(save_dir)


    fragment_location = {}
    for i in tqdm(range(seq.shape[0])):
        if(i%200==0 and i):
            for key in fragment_location.keys():
                utils.save_dict_to_yaml(fragment_location[key], os.path.join(save_dir,f'{key}_fragment_location.yaml'))

        locs = cal_fragment_location_faster(network=sample_network,conv_layer_name=conv_layer_name, 
                                                maxactivations=maxactivations,threshold=thresh_hold,seq=seq[i],
                                                recep_field=recp_field[conv_layer_name],device=device)
        for k in range(neuron_num):
            loc = locs[k]
            task = f'{conv_layer_name}_neuron{k + 1}'
            for j, start in enumerate(loc):
                if task not in fragment_location:
                    fragment_location[task]={}
                # Each sequence is sampled only once.
                if(j > 0):
                    break
                fragment_location[task][f'Seq{i}_{j}'] = start.item()

    
