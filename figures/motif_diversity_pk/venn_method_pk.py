
import numpy as np
import os
import matplotlib.pyplot as plt
import venn
import pandas as pd
from datetime import datetime
import sys
sys.path.append('./')
import utils
import argparse


current_folder = os.path.dirname(os.path.abspath(__file__))

os.chdir(current_folder)

cfgs = utils.read_yaml_to_dict('../../configs/others/method_pk.yaml')

method_list = cfgs['method_list']
model_name = cfgs['model_name']
motif_list = {}

base_dir = '../../motif_discovery/analysis/tomtom_results'
save_dir = './'
conv_layer_list= cfgs['conv_layer_list']
motif_cluster = utils.read_yaml_to_dict('../../data/human_rbp_microrna_motif_0.001.yaml')

if(not os.path.exists(save_dir)):
    os.makedirs(save_dir)
for method_name in method_list:
    df = pd.read_csv(os.path.join(base_dir,method_name,'%s_all_motifs_drop_duplicates.csv'%model_name))
    df = df[df['task'].str.contains('|'.join(conv_layer_list))]
    df['Target_ID_clustrered'] = df['Target_ID'].map(motif_cluster)
    unique_df = df['Target_ID_clustrered'].drop_duplicates()
    motif_list[method_name]= unique_df.to_list()

sets = [set(motif_list[key]) for key in motif_list]
labels = venn.generate_petal_labels(sets)

venn.venn4(labels, names=motif_list.keys(),dpi=200)

plt.title("Venn Diagram")

current_time = datetime.now()

image_filename = current_time.strftime('%Y%m%d%H%M%S') + '_all_motifs_venn.png'

plt.savefig(os.path.join(save_dir,image_filename))