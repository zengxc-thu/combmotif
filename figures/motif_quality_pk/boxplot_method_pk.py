import numpy as np
import os
import matplotlib.pyplot as plt
import venn
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import sys
sys.path.append('./')
import utils


current_folder = os.path.dirname(os.path.abspath(__file__))

os.chdir(current_folder)

cfgs = utils.read_yaml_to_dict('../../configs/others/method_pk.yaml')


method_list = cfgs['method_list']
model_name = cfgs['model_name']
motif_list = {}
base_dir = '../../motif_discovery/analysis/tomtom_results'
save_dir = './'
conv_layer_list= cfgs['conv_layer_list']
if(not os.path.exists(save_dir)):
    os.makedirs(save_dir)
merged_df = pd.DataFrame()
for method_name in method_list:
    df = pd.read_csv(os.path.join(base_dir,method_name,'%s_all_motifs_drop_duplicates.csv'%model_name))
    df = df[df['task'].str.contains('|'.join(conv_layer_list))]
    df = df.loc[df.groupby(['Target_ID'])['q-value'].idxmin()]
    unique_df = df.loc[df.groupby(['Target_ID'])['q-value'].idxmin()].copy()
    unique_df = unique_df[['Target_ID', 'q-value']]
    motif_list[method_name] = unique_df


merged_df = pd.DataFrame()
for method_name in method_list:
    if(len(merged_df)==0):
        merged_df = pd.concat([merged_df,motif_list[method_name]])
        continue

    merged_df = merged_df.merge(motif_list[method_name], left_on='Target_ID', right_on='Target_ID', how='inner')




box = {}
position = {}
for i, method_name in enumerate(method_list):
    box[method_name] = merged_df.iloc[:, i + 1].values
    box[method_name] = box[method_name][np.where(box[method_name] > 0)]
    position[method_name] = box[method_name].copy()
    position[method_name][:] = i + 1



box_list = [box[method_name] for method_name in method_list]


s = np.repeat(method_list, [len(box[method_name]) for method_name in method_list])
value = np.concatenate([box[method_name] for method_name in method_list])

df = pd.DataFrame({'Method': s,
                   '-log(q-value)': -np.log10(value)})

sns.set(style="whitegrid")
ax = sns.boxplot(x='Method', y='-log(q-value)', data=df)

ax.grid(True)

current_time = datetime.now()

image_filename = current_time.strftime('%Y%m%d%H%M%S') + '_box_plot.png'

plt.savefig(os.path.join(save_dir,image_filename))





for i in range(len(method_list)):
    for j in range(i + 1, len(method_list)):
        print(f'{method_list[i]} v.s {method_list[j]}')

        box_1 = box[method_list[i]]
        box_2 = box[method_list[j]]

        p = stats.levene(box_1, box_2)[1]
        if p > 0.05:
            print('Satisfy equal variance')
        w, p = stats.wilcoxon(box_1, box_2)
        print(p)
        if p < 0.05:
            print('p=%.4f:Significant difference in mean' % p)
        else:
            print('p=%.4f:No significant difference in mean ' % p)



