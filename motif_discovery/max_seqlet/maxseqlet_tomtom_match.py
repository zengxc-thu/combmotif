# -*- coding = utf-8 -*-
# @Time : 2022/9/19 20:23
# @Author : zxc
# @Software :基于deeplift的modisco方法 在meme中匹配 一套流程
from __future__ import division, print_function
import os
import joblib
import model_zoo
import datetime
import random
import pandas as pd
import utils
from utils import get_fitness_and_activation_2_0
import re
from math import log10
import numpy as np
import torch

def run_cmd(cmd_str='', echo_print=0):

    from subprocess import run
    if echo_print == 1:
        print('\nExecute cmd instruction="{}"'.format(cmd_str))
    run(cmd_str, shell=True)
def check_rows(matrix):
    # Sum the elements in each row; if the sum equals the length of the row, the entire row is set to 1, return True; otherwise, return False.
    return (np.sum(matrix, axis=1) == matrix.shape[1]).reshape(-1, 1)
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
if __name__ == "__main__":

    
    current_folder = os.path.dirname(os.path.abspath(__file__))

    
    os.chdir(current_folder)

    args = utils.get_explain_args()

    cfgs = utils.read_yaml_to_dict(args.config)
    
    for key in cfgs.keys():
        if(key == 'ind' and cfgs['multi_thread']):
            continue
        args[key] = cfgs[key]

    conv_layer_name = args.name

    ind = re.findall("\d+", args.ind)
    neuron_index_list = np.arange(int(ind[0]), int(ind[1]))
    print('max seqlet information:%s %s' % (conv_layer_name, ind))

    set_random_seed(args.seed)
    dt = datetime.datetime.now()


    repeat = False
    top_fitness_pop_size = 100

    dataset = 'ga' 
    dt = datetime.datetime.now()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sample_network = model_zoo.Saluki_Motif(num_layers = args.num_layers, seq_length = args.seq_length, seq_depth = args.seq_depth, num_targets = args.num_targets)

    sample_network.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model_state_dict'])
    sample_network.to(device)
    sample_network.eval()

    if dataset=='train':
        sample_save_path = os.path.join('../training_maxactivation_samples', args.model_name)
    elif dataset == 'ga':
        sample_save_path = os.path.join('../maxactivation_samples', args.model_name)

    st_lib = args.st_lib
    save_path = './tomtom_match_results/%s/'%args.model_name + '%s'%conv_layer_name + '/{}'.format(
                                                                                dt.date())
    
    utils.create_dir(save_path)

    for neuron_index in neuron_index_list:
        motif_cnt = 0
        neuron_name = 'neuron%s' % (neuron_index + 1)
        task = conv_layer_name + '_' + neuron_name


        if(not os.path.exists(sample_save_path + '/' + conv_layer_name + '_neuron' + str(
            neuron_index + 1) + '.pkl')):
            print(task, f':There is no samples collected. Plz run motif_discovery/multi_thread_script/main.sh first')
            continue

        sample = joblib.load(sample_save_path + '/' + conv_layer_name + '_neuron' + str(
            neuron_index + 1) + '.pkl')
        
        if sample.shape[0] <= 5:
            print(task, ':The samples are not enough,because the current neurons are almost not being activated.')
            continue

        fitness, activation_value = get_fitness_and_activation_2_0(sample.clone(), sample_network, device,
                                                 neuron_index, conv_layer_name)
        

        
        fitness_descending = torch.sort(fitness, descending=True, dim=0)
        top_fitness_indices = fitness_descending.indices[0:top_fitness_pop_size]
        top_samples = sample[top_fitness_indices[:, 0], ...].clone()
        qualified_rows = check_rows(top_samples.sum(dim=1).numpy())
        top_samples = top_samples[qualified_rows[:,0]]

        if(top_samples.shape[0]==0):
            print(task, 'No activated samples.')
            continue


        
        PFM = np.sum(top_samples.numpy(), axis=0)
        CC = PFM.astype(int)
        CC = np.transpose(CC)
        np.savetxt(os.path.join(save_path, f'{task}_PFM.txt'), CC,
                    fmt='%d', delimiter=' ')
        
        print('*********************%s:START tomtom match************************' % task)
        
        match_flag = False
        if not os.path.exists('%s/tomtom_out_%s/tomtom.tsv' % (save_path, task)):
            motif_range = utils.divide_2_0(PFM.copy(),filter = False)
            for j, ind in enumerate(motif_range):
                match_flag = True
                motif_cnt += 1
                motif_i = PFM[:, ind[0]: ind[1] + 1]

                cluster_name = '%d-%dto%d' % (j, ind[0], ind[1])
                print(cluster_name)

                f = open(save_path + '/%s.chen' % task, 'a')

                if motif_i.shape[0] == 4:
                    motif_i = np.transpose(motif_i)
                for index, c in enumerate(motif_i):
                    if index == 0:
                        f.writelines('>%s\n' % cluster_name)
                    f.writelines('%d %d %d %d\n' % (c[0], c[1], c[2], c[3]))
                f.close()
        else:
            print('Skip %s because it has been tomtomed before'%task)
            continue

        
        if(match_flag):
            run_cmd('chen2meme %s/%s.chen > %s/%s.meme' % (save_path, task, save_path, task))
            if(args.alphabet=='RNA'):
                utils.convert_dna_to_rna_meme('%s/%s.meme' % (save_path, task), '%s/%s.meme' % (save_path, task))
            run_cmd('tomtom -thresh %s -norc -o %s/tomtom_out_%s %s/%s.meme %s' % (args.thresh, save_path, task, save_path, task, st_lib))
            run_cmd('rm %s/%s.meme' % (save_path, task))
            run_cmd('rm %s/%s.chen' % (save_path, task))
        else:
            print("No query motif for %s. Current neuron doesn't have any preferred patterns" %task)
            continue
        ##存储最大p值
        stat_res = []
        max_p = -10000
        max_name = None
        with open('%s/tomtom_out_%s/tomtom.tsv' % (save_path, task), 'r') as f2:
            lines = f2.readlines()
            lines = lines[1:-4]
            for line in lines:
                line = line.split('\t')
                p = -log10(float(line[4]))
                name = line[0] + '_' + line[1]
                if p > max_p:
                    max_p = p
                    max_name = name
        if('max_name' in locals()):
            stat_res.append([task, max_name, max_p])
            stat_res_pd = pd.DataFrame(stat_res)
            stat_res_pd.to_csv('%s/p_value_list.csv' % save_path, index=False, mode='a', header=False)
        else:
            print("No valid target motif matched for %s. Current neuron's preferred patterns are not similar to any motif in the library" %task)

        





