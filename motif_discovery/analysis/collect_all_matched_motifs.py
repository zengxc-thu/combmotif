import numpy as np
import os
import pandas as pd
import sys

# The purpose is to collect all motifs matched by the specified model, not just the optimal motif recognized by each neuron. 
# The procedure involves reading each tomtom matching .tsv file and concatenating them together.

current_folder = os.path.dirname(os.path.abspath(__file__))

os.chdir(current_folder)

tomtom_res_base_path = sys.argv[1]

method = sys.argv[2]

model_name = sys.argv[3]

save_dir = f'./tomtom_results/{method}'

if(not os.path.exists(save_dir)):
    os.makedirs(save_dir)

top_folder_path = os.path.join(tomtom_res_base_path, model_name)

layer_name_list = os.listdir(top_folder_path)
# layer_name_list = ['conv3','conv4']

custom_column_names = ['Query_ID', 'Target_ID', 'Optimal_offset', 'p-value', 'E-value',
       'q-value', 'Overlap', 'Query_consensus', 'Target_consensus',
       'Orientation','task']  


combined_df = pd.DataFrame()
for layer in layer_name_list:
    layer_dir = os.path.join(top_folder_path,layer)
    timestamp_list = os.listdir(layer_dir)
    for timestamp in timestamp_list:
        timestamp_dir = os.path.join(layer_dir,timestamp)
        try:
            neuron_list = os.listdir(timestamp_dir)
        except NotADirectoryError:
            continue
        for neuron in neuron_list:
            if('tomtom' not in neuron):
                continue
            neuron_dir = os.path.join(timestamp_dir,neuron)
            tom_tsv_dir = os.path.join(neuron_dir,'tomtom.tsv')
            xml_tsv_dir = os.path.join(neuron_dir,'tomtom.xml')
            task = neuron[len('tomtom_out_'):]
            if os.path.exists(xml_tsv_dir):
                df = pd.read_csv(tom_tsv_dir, delimiter='\t')
                df_trimmed = df.iloc[:-3, :].copy()
                if(len(df_trimmed)==0):
                    print('%s:No motifs were matched.' % (task))
                    continue
                df_trimmed['task'] = task
                combined_df = pd.concat([combined_df, df_trimmed], ignore_index=True,axis=0)

            else:
                print('%s:No motifs were matched.' % (task))

combined_df.to_csv(os.path.join(save_dir,'%s_all_motifs_drop_duplicates.csv'%model_name), index=False)


                

        


