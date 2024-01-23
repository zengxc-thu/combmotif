# -*- coding = utf-8 -*-
# @Time : 2022/9/19 20:23
# @Author : zxc
# @Software :基于deeplift的modisco方法 在meme中匹配 一套流程
from __future__ import division, print_function
import os
from collections import OrderedDict
import joblib
from importlib import reload
import time
import model_zoo
import datetime
import random
import utils
from utils import (get_region, generate_initial_group,
    get_gredient_2_0, probability_transform_3_0, mutation_13_0,
    cyclic_shift_2_0, cross_over_5_0, reinitial_same_pop, seq_to_name, 
    get_fitness_and_activation_2_0,get_featuremap_for_cluster,cluster_name_to_file_path,
    record_pfm_id_info)
import re
import h5py
from math import log10
import numpy as np
import torch
def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"文件 {file_path} 已成功删除。")
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
    except Exception as e:
        print(f"删除文件时发生错误: {e}")
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
    print('tfmodisco information:%s %s ' % (conv_layer_name, ind))

    set_random_seed(args.seed)
    dt = datetime.datetime.now()


    sense_field = utils.read_yaml_to_dict(args.recp_path)

    repeat = False

    mode = 'modisco'  ##主要有两种 ga_modesico(ga采样 modisco解析)还有modisco(直接从训练集采样 modisco解析)
    dataset = 'ga'  ## 还有train

    modisco_save_dir = os.path.join('%s_results'%mode, args.model_name) ##注意修改 ga代表采样 modisco代表聚类

    dt = datetime.datetime.now()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ## 初始化网络
    sample_network = model_zoo.Saluki_Motif(num_layers = args.num_layers, seq_length = args.seq_length, seq_depth = args.seq_depth, num_targets = args.num_targets)

    sample_network.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model_state_dict'])
    sample_network.to(device)
    sample_network.eval()

    if dataset=='train':
        sample_save_path = '../training_maxactivation_samples'
    elif dataset == 'ga':
        sample_save_path = os.path.join('../maxactivation_samples', args.model_name)

    utils.create_dir(modisco_save_dir)

    st_lib = '../../data/human_rbp_microrna_motif.meme'
    save_path = os.path.join('stats', args.model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    window = sense_field[conv_layer_name]

    for neuron_index in neuron_index_list:
        motif_cnt = 0
        empty_flag = False
        ideal_flag = True  ##节约时间 有些神经元根本不激活 采不到样本 没必要
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

        if sample.shape[2] == 4:
            sample = sample.permute([0, 2, 1])
        

        print('*********************%s:START modisco************************' % task)


        outfile = os.path.join(modisco_save_dir, f"{task}_tfmodisco.h5")
        
        if not os.path.exists(outfile) or repeat:

            gredient = get_gredient_2_0(sample.clone(), sample_network, device, neuron_index,conv_layer_name, batch_cal=True)
            gredient = gredient.cpu() * sample

            onehot_data = sample.cpu().numpy()

            x_npz_path = os.path.join(save_path, f"{task}_x.npz")
            grad_npz_path = os.path.join(save_path, f"{task}_grad.npz")
            np.savez(x_npz_path, sample.numpy())
            np.savez(grad_npz_path, gredient.numpy())

            n_seqlet = 5000

            outfile = os.path.join(modisco_save_dir, f"{task}_tfmodisco.h5")
            
            command = f"modisco motifs -s {x_npz_path} -a {grad_npz_path} -n {n_seqlet} -o {outfile} -w {window}"
            
            run_cmd(command)

            with h5py.File(outfile, "r") as f:
                print(f.keys())
                if 'pos_patterns' in f.keys():
                    pos_motif = f['pos_patterns'] 
                    n_pos = len(pos_motif.keys())
                else:
                    n_pos = -1

                if 'neg_patterns' in f.keys():
                    neg_motif = f['neg_patterns'] 
                    n_neg = len(neg_motif.keys())
                else:
                    n_neg = -1
                f.close()

                print(f"discover {n_pos} pos pattern and {n_neg} neg patterns for n = {n_seqlet}  w = {window} ")
            delete_file(x_npz_path)
            delete_file(grad_npz_path)
        else:
            print('*********************%s:already tfmodiscoed************************' % task)



