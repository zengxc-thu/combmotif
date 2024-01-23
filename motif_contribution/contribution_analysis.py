# -*- coding = utf-8 -*-
# @Time : 2023/8/16 8:20
# @Author : zxc
# @Software : 
from __future__ import division, print_function
import os
import joblib
import utils
import h5py
import torch
import pandas as pd
from scipy.stats import ttest_ind



def ttest(a, b, alpha=0.05):
    t_statistic, p_value = ttest_ind(a, b)
    # Determine whether to reject the original hypothesis based on p-value (equal means)
    alpha = 0.05 
    if p_value < alpha:
        # print("Rejecting the null hypothesis: the mean values of two sets of samples are not equal")
        if (a.mean() > b.mean()):
            return p_value, 1
        else:
            return p_value, -1
    else:
        # print("Accept the null hypothesis: the mean values of two sets of samples are equal")
        return 0, 0


if __name__ == "__main__":
    
    current_folder = os.path.dirname(os.path.abspath(__file__))

    
    os.chdir(current_folder)
    args = utils.get_explain_args()

    cfgs = utils.read_yaml_to_dict(args.config)
    for key in cfgs.keys():
        args[key] = cfgs[key]

    target_name = 'hl'

    print(f'\nStart merging identification results with half-life label of ({args.data_path})...')

    file = h5py.File(args.data_path, 'r')

    train_label = file['train_label'] 
    train_label =  torch.from_numpy(train_label[:]).float()
    valid_label = file['valid_label'] 
    valid_label = torch.from_numpy(valid_label[:]).float()  
    test_label = file['test_label'] 
    test_label = torch.from_numpy(test_label[:]).float() 
    if(len(test_label.shape)==1):
        label = torch.cat((train_label, valid_label, test_label), dim=0) 
        df = pd.DataFrame({target_name:label.ravel()})
    else:
        df = pd.DataFrame()
        label = torch.cat((train_label, valid_label, test_label),dim=0) 
        for i in range(test_label.shape[1]):
            df[f'hl{i}'] = label[:, i]

    threshold_list=[i/10 for i in range(1, 10)]

    threshold = 0.7

    ind = threshold_list.index(threshold)
    sample_label_dir = './seq_act_label/%s/%s'%(args.model_name,threshold_list) 
    save_dir = './results'
    # sample_annotations_dir = './neuron_annotation_dataset/%s/threshold_%s'%(args.model_name,threshold)   
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    items = os.listdir(sample_label_dir)

    for item in items:
        item_path = os.path.join(sample_label_dir, item)
        conv_layer_name = item.split('.')[0]
        sample_label = joblib.load(item_path)
        assert sample_label.shape[0]==len(df)
        # We assume that the default `sample_label` includes all neurons.
        # sample_label.shape = [#samples, #neurons, ##thresholds]
        for neuron_index in range(sample_label.shape[1]):
            df[conv_layer_name + '_neuron%d'%(neuron_index+1)]=sample_label[:,neuron_index,ind]


    # df.to_csv(os.path.join(sample_annotations_dir,'%s_alldataset_annotation.csv'%args.model_name),index=False)
    # print(f"Final merging results are saved to {os.path.join(sample_annotations_dir,'%s_alldataset_annotation.csv'%args.model_name)}")

    print(f'\nStart compute the contribution of each neuron to the final prediction ...')
    columns = [col for col in df.columns if 'conv' in col]


    positive_columns = []
    negative_columns = []

    motif_contr = pd.DataFrame(columns=['task', 'contribution','p-value','#neuron-activating samples','#neuron-inactive samples'])


    for task in columns:

        seq_act = df[df[task] == 1.0]
        seq_notact = df[df[task] == 0.0]

        contribution_type = 'none'
        p_value = -1
        act_num = len(seq_act)
        noact_num = len(seq_notact)


        if (act_num == 0 or noact_num == 0):
            new_row = {'task': task, 'contribution': contribution_type, 'p-value':p_value,'#neuron-activating samples':act_num,'#neuron-inactive samples':noact_num}
            motif_contr = motif_contr.append(new_row, ignore_index=True)
            continue


        p_value, flag = ttest(seq_act[target_name], seq_notact[target_name])
        if (flag == 0):
            contribution_type = 'neutral'
        elif (flag == 1):
            contribution_type = 'positive'
        elif (flag == -1):
            contribution_type = 'negative'

        new_row = {'task': task, 'contribution': contribution_type, 'p-value':p_value,'#neuron-activating samples':act_num,'#neuron-inactive samples':noact_num}
        motif_contr = motif_contr.append(new_row, ignore_index=True)

    motif_contr.to_csv(os.path.join(save_dir,'%s_neuron_contribution.csv'%args.model_name),index=False)
    print(f"Final neuron contribution results are saved to {os.path.join(save_dir,'%s_neuron_contribution.csv'%args.model_name)}")
