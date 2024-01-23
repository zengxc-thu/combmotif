




# -*- coding = utf-8 -*-
# @Time : 2023/9/26 15:13
# @Author : zxc
# @Software :
from __future__ import division, print_function
import os
import re
import joblib
import sys
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import utils
import datetime
from model_zoo import Saluki_Motif,saluki_torch
import h5py
from math import log10
import numpy as np
import random
import torch
import pandas as pd
import sys
from scipy.stats import ttest_ind
from scipy.stats import wilcoxon
from utils import get_activation,get_gredient_2_0
def get_first_number_after_seq(input_string):
    # Use a regular expression to match the first number after 'seq'.
    match = re.search(r'Seq(\d+)', input_string)

    # If a matching item is found, return the matched number.
    if match:
        return int(match.group(1))
    else:

        return None
# def contains_any_substring(input_string, substring_list):
#     return any(substring in input_string for substring in substring_list)
def check_dna_pattern(dna_sequence, pattern):

    
    pattern = pattern.replace('N', '.')

    match = re.search(pattern, dna_sequence)
    
    return bool(match)


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

def cal_fragment_location(network,conv_layer_name, neuron_index, maxactivation, seq, recep_field,threshold,device):

    assert(seq.shape[1] >= recep_field)

    samples = torch.zeros(seq.shape[1] - recep_field + 1, seq.shape[0], recep_field)
    for i in range(seq.shape[1] - recep_field + 1):
        samples[i] = seq[:,i:i+recep_field]


    activation_value = get_activation(samples,network,conv_layer_name,device)

    activation_value = activation_value[:, neuron_index]
    

    return torch.where(activation_value>threshold*maxactivation)[0]


def find_positions(s, motif):
    
    positions = {}
    
    motif = motif.replace('N', '.')

    matches = re.finditer(motif, s)

    positions[motif] = [match.start() for match in matches]

    return positions

def motif_perbulation(network, whole_seq, a_start,a_end, b_start,b_end):
    sample = torch.tile(whole_seq,(4,1,1))
    astr = utils.one_hot_to_dna(sample[0,:4, a_start:a_end+1].numpy())
    bstr = utils.one_hot_to_dna(sample[0,:4, b_start:b_end+1].numpy())
    print(f"Modify two motif:{astr} and {bstr}")
    

    for i in range(1, 4):
        if(i==1):
            # scr a
            sample[i,:4, a_start:a_end+1] = utils.generate_initial_group(1, total_length=a_end - a_start + 1)[0]
        elif(i==2):
            # scr b
            sample[i,:4, b_start:b_end+1] = utils.generate_initial_group(1, total_length=b_end - b_start + 1)[0]
        elif(i==3):
            # scr a+b
            sample[i,:4, b_start:b_end+1] = utils.generate_initial_group(1, total_length=b_end - b_start + 1)[0]
            sample[i,:4, a_start:a_end+1] = utils.generate_initial_group(1, total_length=a_end - a_start + 1)[0]

        astr = utils.one_hot_to_dna(sample[i,:4, a_start:a_end+1].numpy())
        bstr = utils.one_hot_to_dna(sample[i,:4, b_start:b_end+1].numpy())
        print(f"After modifying:{astr} and {bstr}")
    pred = network(sample)
    return pred[:,args.reference_label]








if __name__ == "__main__":
    
    current_folder = os.path.dirname(os.path.abspath(__file__))

    
    os.chdir(current_folder)

    args = utils.get_explain_args()

    cfgs = utils.read_yaml_to_dict(args.config)
    for key in cfgs.keys():
        args[key] = cfgs[key]

    conv_layer_name = args.name
    ind = re.findall("\d+", args.ind)
    neuron_index_list = np.arange(int(ind[0]), int(ind[1]))


    set_random_seed(args.seed)
    dt = datetime.datetime.now()


    
    recp_field = utils.read_yaml_to_dict(args.recp_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize model for calculation neuron activation 
    sample_network = Saluki_Motif(num_layers=args.num_layer,
                                   seq_depth = args.seq_depth, 
                                   num_targets = args.num_targets)
    sample_network.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model_state_dict'])
    sample_network.to(device)
    sample_network.eval()

    # initialize model for calculation half life
    model = saluki_torch(num_layers=args.num_layer,
                        seq_depth = args.seq_depth, 
                        num_targets = args.num_targets)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model_state_dict'])
    model.eval()



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
    seq2 = seq.clone()
    seq2[:,4:,:]=0#Avoid the impact of splicing and coding frame
    seq = seq[:,:4,:]
    label = torch.cat((train_label, valid_label, test_label), dim=0)

    df_all = pd.DataFrame(columns=['task','motifA','motifB','a-o','b-o','ab-a','ab-b','ab-o','Wilcoxon-p'])

    for neuron_index in neuron_index_list:

        neuronname = 'neuron' + str(neuron_index + 1)
        task = conv_layer_name + '_' + neuronname
        print('*********************%s:START Scrambling************************' % task)
        combination_dir = os.path.join(args.combination, args.model_name, f'{task}.yaml')
        fragment_location_dir = os.path.join(args.fragment, args.model_name, f'{task}_fragment_location.yaml')
    
        
        if(os.path.exists(fragment_location_dir)):
            fragment_location = utils.read_yaml_to_dict(fragment_location_dir)
        else:
            print(f"{fragment_location_dir} doesn't exist")
            print(f"{task} : There are no sequences collected reaching the activation threshold.")
            continue
        if(not fragment_location):
            print(f"{task} : There are no sequences collected reaching the activation threshold.")
            continue

        sample = torch.zeros([len(fragment_location), seq.shape[1], recp_field[conv_layer_name]])
        whole_seq_sample = torch.zeros([len(fragment_location), seq2.shape[1], seq.shape[2]])
        assert(sample.shape[1] == 4)

        start_end_list = [] #Inclusive on the left, inclusive on the right.

        for i, key in enumerate(fragment_location.keys()):
            seq_id = get_first_number_after_seq(key)
            start = fragment_location[key]
            sample[i] = seq[seq_id,:,start:start+recp_field[conv_layer_name]]
            whole_seq_sample[i] = seq2[seq_id] # We need the complete sequence, including splicing.
            # astr = utils.one_hot_to_dna(sample[i].numpy())
            # bstr = utils.one_hot_to_dna(whole_seq_sample[i,:4,start:start+recp_field[conv_layer_name]].numpy())
            # assert(astr==bstr)
            start_end_list.append([start, start+recp_field[conv_layer_name] - 1])

        activation_value = get_activation(sample,sample_network,conv_layer_name,device)
        activation_value = activation_value[:, neuron_index]
        print(f'Please check if the activation values for each sequence with respect to [ {conv_layer_name}_neuron{neuron_index + 1}] are reasonable:\n',activation_value.tolist()[:10])

        dna_sequences = utils.one_hot_to_dna(sample.numpy())

        if(not os.path.exists(combination_dir)):
            combinations = utils.read_yaml_to_dict(args.template)
            print(f'**************Plz define the motif combination in motif_interaction/motif_combination_labels/hl_predictor**************')
            print(f'**************Temporarily using template combination from {combinations}**************')
        else:
            combinations = utils.read_yaml_to_dict(combination_dir)


        for key in combinations.keys():
            print(f"**************{key}***************")

            motifA = combinations[key]['motifA'][0]
            motifB = combinations[key]['motifB'][0]

            target_ind = []
            sequence_with_motif = []

            save_dir = f'results/scramble_res/{args.model_name}/{task}/{motifA}_{motifB}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            



            dic = {}
            with open(os.path.join(save_dir , 'high_act_fragments.txt'), 'w') as file:
                for ind, sequence in enumerate(dna_sequences):
                    file.write(sequence + '\n')    
                    if check_dna_pattern(sequence, motifA) and check_dna_pattern(sequence, motifB):
                        
                        target_ind.append(ind)
                        sequence_with_motif.append(sequence)

                        temp_dic = {}
                        temp_dic['motifA'] = find_positions(sequence, motifA)
                        temp_dic['motifB'] = find_positions(sequence, motifB)

                        dic[f'seq{ind}'] = temp_dic
            df = pd.DataFrame(columns=['original','scr a','scr b','scr a+b'])
            
            for seq_name in dic.keys():
                seq_id = int(seq_name.split('seq')[-1])

                motifA_name = list(dic[seq_name]['motifA'].keys())[0]
                motifA_len = len(motifA_name)
                motifA_start_list = dic[seq_name]['motifA'][motifA_name]

                motifB_name = list(dic[seq_name]['motifB'].keys())[0]
                motifB_len = len(motifB_name)
                motifB_start_list = dic[seq_name]['motifB'][motifB_name]

                for a_start in motifA_start_list:
                    for b_start in motifB_start_list:
                        if(np.abs(a_start-b_start)<max(motifA_len, motifB_len)):continue
                        # a_start += start_end_list[seq_id][0]
                        # b_start += start_end_list[seq_id][0]
                        # print(utils.one_hot_to_dna(sample[seq_id, :4, a_start : a_start + motifA_len].numpy()), end=" ")
                        # print(utils.one_hot_to_dna(sample[seq_id, :4, b_start : b_start + motifB_len].numpy()))
                        start = start_end_list[seq_id][0]

                        # astr = utils.one_hot_to_dna(whole_seq_sample[seq_id,:4,start+a_start:start+a_start + motifA_len].numpy())
                        # bstr = utils.one_hot_to_dna(whole_seq_sample[seq_id,:4,start+b_start:start+b_start + motifB_len].numpy())
                        # print(f"modify motif:{astr} and {bstr}")
                        pred = motif_perbulation(network=model, whole_seq=whole_seq_sample[seq_id], 
                                        a_start = start + a_start, a_end=start + a_start + motifA_len - 1,
                                        b_start = start + b_start, b_end=start + b_start + motifB_len - 1)
                        



                        data = pred.tolist()
                        df = df.append(pd.Series(data, index=df.columns), ignore_index=True)

            if(len(df)<=5):continue
            plt.clf()
            sns.set(style="whitegrid")
            statistic, p_value_ao = wilcoxon(df['scr a'], df['original'])
            statistic, p_value_bo = wilcoxon(df['scr b'], df['original'])
            statistic, p_value_ab_a = wilcoxon(df['scr a+b'], df['scr a'])
            statistic, p_value_ab_b = wilcoxon(df['scr a+b'], df['scr b'])
            statistic, p_value_ab_o = wilcoxon(df['scr a+b'], df['original'])

            plt.figure(figsize=(10, 8))
            sns.boxplot(data=df)
            
            plt.text(0.5, 0.95, f'p-value (a vs o): {p_value_ao:.4f}', ha='center', va='center', transform=plt.gca().transAxes)
            plt.text(0.5, 1.0, f'p-value (b vs o): {p_value_bo:.4f}', ha='center', va='center', transform=plt.gca().transAxes)
            plt.text(0.5, 1.05, f'p-value (ab vs a): {p_value_ab_a:.4f}', ha='center', va='center', transform=plt.gca().transAxes)
            plt.text(0.5, 1.1, f'p-value (ab vs b): {p_value_ab_b:.4f}', ha='center', va='center', transform=plt.gca().transAxes)
            plt.text(0.5, 0.9, f'p-value (ab vs o): {p_value_ab_o:.4f}', ha='center', va='center', transform=plt.gca().transAxes)
            plt.savefig(os.path.join(save_dir,f'boxplot.png'))
        

            if(df['scr a'].mean()>df['original'].mean()):p_value_ao *= -1
            if(df['scr b'].mean()>df['original'].mean()):p_value_bo *= -1
            if(df['scr a+b'].mean()>df['scr a'].mean()):p_value_ab_a *= -1
            if(df['scr a+b'].mean()>df['scr b'].mean()):p_value_ab_b *= -1
            if(df['scr a+b'].mean()>df['original'].mean()):p_value_ab_o *= -1


            
            # Consider only positive-positive or negative-negative cases.
            p_value = 0
            type = 'none'
            if(not (p_value_bo*p_value_ao<0 and np.abs(p_value_bo)<0.05 and np.abs(p_value_ao)<0.05)):
                df['a+b-2n'] = df['scr a'] + df['scr b'] - 2 * df['scr a+b']  #the sum of the marginal effect sizes
                df['o-n'] = df['original'] - df['scr a+b'] # the joint effect size
                statistic, p_value = wilcoxon(df['o-n'], df['a+b-2n'])
                if(p_value_ao > 0):
                    # positive motif

                    # o-n>a+b-2n positive synergistic
                    # o-n=a+b-2n addictive
                    # o-n<a+b-2n positive antagonistic
                    if(df['o-n'].mean() > df['a+b-2n'].mean() and p_value < 0.05):
                        type = 'positive_synergistic'
                    elif(df['o-n'].mean() < df['a+b-2n'].mean() and p_value < 0.05):
                        type = 'positive_antagonistic'
                    elif(p_value >= 0.05):
                        type = 'addictive'
                else:
                    # negative motif

                    # o-n<a+b-2n negative synergistic
                    # o-n=a+b-2n addictive
                    # o-n>a+b-2n negative antagonistic
                    if(df['o-n'].mean() < df['a+b-2n'].mean() and p_value < 0.05):
                        type = 'negative_synergistic'
                    elif(df['o-n'].mean() > df['a+b-2n'].mean() and p_value < 0.05):
                        type = 'negative_antagonistic'
                    elif(p_value >= 0.05):
                        type = 'addictive'

                

            new_data = {'task': task, 'motifA': motifA, 'motifB': motifB, 'a-o': p_value_ao, 'b-o': p_value_bo, 'ab-a': p_value_ab_a, 'ab-b': p_value_ab_b, 'ab-o':p_value_ab_o,'interaction-p':p_value,'interaction':type}
            df_all = df_all.append(new_data, ignore_index=True)
    

    stats_save_dir = f'./results/stats/{args.model_name}'
    if(not os.path.exists(stats_save_dir)):
        os.makedirs(stats_save_dir)
    df_all.to_csv(os.path.join(stats_save_dir, f'motif_scramble_stats.csv'),index=False)




                        



