# -*- coding = utf-8 -*-
# @Time : 2022/9/19 20:23
# @Author : zxc
# @Software :基于deeplift的modisco方法 在meme中匹配 一套流程
from __future__ import division, print_function
import os
import pandas as pd
import model_zoo
import datetime
import random
import utils
import re
import h5py
from math import log10
import numpy as np
import torch
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


    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    conv_layer_name = args.name
    ind = re.findall("\d+", args.ind)
    neuron_index_list = np.arange(int(ind[0]), int(ind[1]))
    print('information:%s %s' % (conv_layer_name, ind))

    set_random_seed(args.seed)

    mode = 'modisco'  ##主要有两种 ga_modesico(ga采样 modisco解析)还有modisco(直接从训练集采样 modisco解析)

    modisco_save_dir = os.path.join('%s_results'%mode, args.model_name) ##注意修改 ga代表采样 modisco代表聚类

    dt = datetime.datetime.now()

    st_lib = '../../data/human_rbp_microrna_motif.meme'
    save_path = './tomtom_match_results/%s/'%args.model_name + '%s'%conv_layer_name + '/{}'.format(
                                                                                   dt.date())
    utils.create_dir(save_path)
    for neuron_index in neuron_index_list:
        motif_cnt = 0

        neuron_name = 'neuron%s' % (neuron_index + 1)
        task = conv_layer_name + '_' + neuron_name

        print('*****************%s: tomtom match ******************' % task)

        if(not os.path.exists(f"{modisco_save_dir}/{task}_tfmodisco.h5")):
            print(f"Not found modsico results of %s in {modisco_save_dir}/{task}_tfmodisco.h5. Plz check the path" %task)
            continue

        hdf5_results = h5py.File(f"{modisco_save_dir}/{task}_tfmodisco.h5" , "r")

        motif_pattern = {}
        for activity in list(hdf5_results.keys()):
            for pattern in list(hdf5_results[activity].keys()):
                motif_pattern[f'{activity}_{pattern}'] = hdf5_results[activity][pattern]['sequence'][:]
        
        if(len(motif_pattern)==0):
            print("No query motif for %s. Current neuron doesn't have any preferred patterns" %task)
            continue


        match_flag = False
        if not os.path.exists('%s/tomtom_out_%s/tomtom.tsv' % (save_path, task)):
            for pattern_name in motif_pattern.keys():
                print(pattern_name)
                info = motif_pattern[pattern_name]
                pwm = info / info.sum(axis=1).reshape(info.sum(axis=1).shape[0], 1)
                back = np.zeros_like(pwm)
                pfm = np.floor(pwm * 1000)
                back[:, 0] = 1000 - np.floor(pwm * 1000).sum(axis=1)
                pfm = pfm + back

                info = np.transpose(info)
                pfm = np.transpose(pfm)

                motif_range = utils.divide_2_0(info.copy())
            
                for j, ind in enumerate(motif_range):
                    match_flag = True
                    motif_cnt += 1

                    motif_i = pfm[:, ind[0]: ind[1] + 1]
                    cluster_name = f'{pattern_name}.{j}-{ind[0]}to{ind[1]}' 
                    print(cluster_name)

                    f = open('%s/%s.chen' % (save_path, task), 'a')

                    if motif_i.shape[0] == 4:
                        motif_i = np.transpose(motif_i)
                    for index, c in enumerate(motif_i):
                        if index == 0:
                            f.writelines('>%s\n' % cluster_name)
                        f.writelines('%d %d %d %d\n' % (c[0], c[1], c[2], c[3]))
                    f.close()
        else:
            print('Skip %s because it has been tomtomed before'%task)
            hdf5_results.close()
            continue

        hdf5_results.close()

        if(match_flag):
            run_cmd('chen2meme %s/%s.chen > %s/%s.meme' % (save_path, task, save_path, task))
            if(args.alphabet=='RNA'):
                utils.convert_dna_to_rna_meme('%s/%s.meme' % (save_path, task), '%s/%s.meme' % (save_path, task))
            run_cmd('tomtom -thresh %s -norc -o %s/tomtom_out_%s %s/%s.meme %s' % (args.thresh, save_path, task, save_path, task, args.st_lib))
            run_cmd('rm %s/%s.chen' % (save_path, task))
            run_cmd('rm %s/%s.meme' % (save_path, task))
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
