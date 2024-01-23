import numpy as np
import os
import matplotlib.pyplot as plt
import venn
from matplotlib_venn import venn2
import pandas as pd
from datetime import datetime
import sys
sys.path.append('./')
import utils



current_folder = os.path.dirname(os.path.abspath(__file__))


os.chdir(current_folder)

cfgs = utils.read_yaml_to_dict('../../configs/others/model_pk.yaml')

method = cfgs['method']
model_list = cfgs['model_list']


motif_list = {}

base_dir = f'../../motif_discovery/analysis/tomtom_results/{method}'
save_dir = './'
motif_cluster = utils.read_yaml_to_dict('../../data/human_rbp_microrna_motif_0.001.yaml')



if(not os.path.exists(save_dir)):
    os.makedirs(save_dir)
for model_name in model_list:
    df = pd.read_csv(os.path.join(base_dir,'%s_all_motifs_drop_duplicates.csv'%model_name))
    df['Target_ID_clustrered'] = df['Target_ID'].map(motif_cluster)
    unique_df = df['Target_ID_clustrered'].drop_duplicates()
    motif_list[model_name]= unique_df.to_list()


sets = [set(motif_list[key]) for key in motif_list]
labels = venn.generate_petal_labels(sets)

venn.venn2(labels, names=model_list,dpi=300)


current_time = datetime.now()


image_filename = current_time.strftime('%Y%m%d%H%M%S') + '_all_motifs_venn.png'

plt.savefig(os.path.join(save_dir,image_filename))