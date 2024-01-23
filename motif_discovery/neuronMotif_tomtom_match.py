# -*- coding = utf-8 -*-
# @Time : 2024/1/14 16:56
# @Author : zxc
# @File : neuronMotif_tomtom_match.py
# @Software :tomtom

import joblib
import pandas as pd
import sys
import os
import datetime
import utils
import numpy as np
from math import log10
import re
def run_cmd(cmd_str='', echo_print=0):

    from subprocess import run
    if echo_print == 1:
        print('\nExecute cmd instruction="{}"'.format(cmd_str))
    run(cmd_str, shell=True)

if __name__ == "__main__":
    
    current_folder = os.path.dirname(os.path.abspath(__file__))

    
    os.chdir(current_folder)

    args = utils.get_explain_args()

    cfgs = utils.read_yaml_to_dict(args.config)
    
    for key in cfgs.keys():
        args[key] = cfgs[key]

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    conv_layer_name = args.name
    ind = re.findall("\d+", args.ind)
    neuron_index_list = np.arange(int(ind[0]),int(ind[1]))
    print(f"\nTomtom for model: {args.model_name}")
    print(f"Tomtom info: comparing queries from : {conv_layer_name}_neuron{neuron_index_list[0] + 1} ... {conv_layer_name}_neuron{neuron_index_list[-1] + 1}\n")


    dt = datetime.datetime.now()

    mode = 'ga'
    thresh = args.thresh
    n_motifs = 50000
    # neuron_index_list = [21]



    for neuron_index in neuron_index_list:

        neuronname = 'neuron' + str(neuron_index + 1)
        task = conv_layer_name + '_' + neuronname

        print('*********************%s:START tomtom match************************' % task)
        motif_cnt = 0

        save_path = './tomtom_match_results/%s/'%args.model_name + '%s'%conv_layer_name + '/{}'.format(
                                                                                   dt.date())

        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
            except FileExistsError:
                print(f"{save_path} exists, won't be created again")

            

        ALL_PFM_PATH = './%s_pfm/%s/%s_all_PFM.pkl' % (mode,args.model_name,task)

        if(os.path.exists(ALL_PFM_PATH)):
            all_PFM = joblib.load(ALL_PFM_PATH)
        else:
            print('%s_all_PFM.pkl not found, skip.'%task)
            print(f'1. Maybe you have not collect and cluser samples for {task}, plz run neuronMotif_adaptive_sample_cluster.py first')
            print(f"2. Maybe there are no samples to activate the {task}.")
            continue

        match_flag = False
        if not os.path.exists('%s/tomtom_out_%s/tomtom.tsv' % (save_path, task)):
            for i, PFM in enumerate(all_PFM):

                motif_range = utils.divide_2_0(PFM.copy(),filter = False)
                # print(motif_range)

                for j, ind in enumerate(motif_range):
                    match_flag = True
                    motif_cnt += 1
                    motif_i = PFM[:, ind[0]: ind[1] + 1]

                    cluster_name = '%d.%d-%dto%d' % (i, j, ind[0], ind[1])
                    # print(cluster_name)

                    f = open(save_path + '/%s.chen' % task, 'a')

                    if motif_i.shape[0] == 4:
                        motif_i = np.transpose(motif_i)
                    for index, c in enumerate(motif_i):
                        if index == 0:
                            f.writelines('>%s\n' % cluster_name)
                        f.writelines('%d %d %d %d\n' % (c[0], c[1], c[2], c[3]))
                    f.close()

                    if motif_cnt >= n_motifs:
                        break
                if motif_cnt >= n_motifs:
                    break
        else:
            print('Skip %s because it has been tomtomed before'%task)
            continue

        if(match_flag):
            run_cmd('chen2meme %s/%s.chen > %s/%s.meme' % (save_path, task, save_path, task))
            if(args.alphabet=='RNA'):
                utils.convert_dna_to_rna_meme('%s/%s.meme' % (save_path, task), '%s/%s.meme' % (save_path, task))
            run_cmd('tomtom -thresh %s -norc -o %s/tomtom_out_%s %s/%s.meme %s' % (thresh, save_path, task, save_path, task, args.st_lib))
            run_cmd('rm %s/%s.chen' % (save_path, task))
            run_cmd('rm %s/%s.meme' % (save_path, task))
        else:
            print("No query motif for %s. Current neuron doesn't have any preferred patterns" %task)
            continue
        ##存储最大p值
        stat_res = []
        max_p = -10000
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
