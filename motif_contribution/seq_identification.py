# -*- coding = utf-8 -*-
# @Time : 2023/7/25 20:23
# @Author : zxc
# @Software :
# @Description :
# For a given neuron, if a segment of a sample in the training set can activate it to 
# achieve more than half of its maximum activation value, then that sample is considered 
# a positive sample. Conversely, it is considered a negative sample if it doesn't meet this
# criterion. Check if there is a difference in Mean Relative Logit (MRL) between these 
# two groups of samples.

from __future__ import division, print_function
import os
import re
import joblib
import utils
import datetime
import model_zoo
import h5py
import numpy as np
import random
import torch
import sys
from tqdm import tqdm
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

def test_activation(network,conv_layer_name,maxactivations, threshhold_list, seq, sample_len,device):

    # label.shape [neuron_num,threshhold_num]
    # "label[i, j]" represents whether the current sequence, when input to the neural network,
    #  can cause the activation value of the i-th neuron to be greater than the j-th threshold.

    label = torch.zeros(maxactivations.ravel().shape[0], len(threshhold_list))
    if(seq.shape[1] < sample_len):
        # The situation where the receptive field is larger than the training sample sequence length.
        samples = torch.zeros(sample_len-seq.shape[1] + 1,seq.shape[0],sample_len)
        for i in range(sample_len-seq.shape[1] + 1):
            samples[i,:,i:i+seq.shape[1]] = seq
    else:
        samples = torch.zeros(seq.shape[1] - sample_len + 1, seq.shape[0], sample_len)
        for i in range(seq.shape[1] - sample_len + 1):
            samples[i] = seq[:,i:i+sample_len]


    activation_value = get_activation(samples,network,conv_layer_name,device)

    
    for ind, threshold in enumerate(threshhold_list):
        label[:,ind]=torch.any(activation_value>threshold*maxactivations.ravel(), axis=0).int()
    return label


if __name__ == "__main__":
    
    current_folder = os.path.dirname(os.path.abspath(__file__))

    
    os.chdir(current_folder)
    args = utils.get_explain_args()

    cfgs = utils.read_yaml_to_dict(args.config)
    for key in cfgs.keys():
        args[key] = cfgs[key]

    conv_layer_name = args.name


    print(f'Start identifying whether each seq in dataset({args.data_path})  \n can activate each neuron in {conv_layer_name} of model ({args.model_name}) or not...')

    set_random_seed(args.seed)
    dt = datetime.datetime.now()

    sense_field = utils.read_yaml_to_dict(args.recp_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_network = model_zoo.Saluki_Motif(num_layers=args.num_layer,
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
    train_label =  torch.from_numpy(train_label[:]).unsqueeze(1).float()
    valid_label = file['valid_label'] 
    valid_label = torch.from_numpy(valid_label[:]).unsqueeze(1).float()  
    test_label = file['test_label'] 
    test_label = torch.from_numpy(test_label[:]).unsqueeze(1).float()  

    seq = torch.cat((train_sequence, valid_sequence, test_sequence), dim=0)
    label = torch.cat((train_label, valid_label, test_label), dim=0)
    seq = seq[:,:4,:]



    sample_save_path = './maxact_from_trainSet/%s'%args.model_name
    threshold_list=[i/10 for i in range(1, 10)]
    maxactivation_save_path = sample_save_path + '/' + conv_layer_name + '_maxactivation.pkl'
    save_dir = './seq_act_label/%s/%s'%(args.model_name,threshold_list)  


    if(not os.path.exists(save_dir)):
        os.makedirs(save_dir) 
    if(not os.path.exists(maxactivation_save_path)):
        print('*********No %s maxactivations found, plz run motif_contribution/script/run_search_maxact.sh first**********' %conv_layer_name)
        sys.exit()

    maxactivations = joblib.load(maxactivation_save_path)
    maxactivations = maxactivations.to(device)

    neuron_num = maxactivations.ravel().shape[0]

    all_labels = torch.zeros(seq.shape[0],neuron_num,len(threshold_list))

    if(not os.path.exists(os.path.join(save_dir, '%s.pkl'%conv_layer_name))):
        for index in tqdm(range(seq.shape[0])):

            # Judging sequentially for each sequence whether it activates all neurons to reach a certain threshold.
            all_labels[index,...]=test_activation(network=sample_network,conv_layer_name=conv_layer_name,
                                                    maxactivations=maxactivations,threshhold_list=threshold_list,seq=seq[index],
                                                    sample_len=sense_field[conv_layer_name],device=device)
        print(f"Results are saved to {os.path.join(save_dir, '%s.pkl'%conv_layer_name)}")
        joblib.dump(all_labels, os.path.join(save_dir, '%s.pkl'%conv_layer_name))
    else:
        print(f"Neurons in {conv_layer_name} have been identified before, continue...")
        all_labels = joblib.load(os.path.join(save_dir, '%s.pkl'%conv_layer_name))

