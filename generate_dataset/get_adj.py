import numpy as np
import sys
import pandas as pd
import os
import joblib
def run_cmd(cmd_str='', echo_print=0):

    from subprocess import run
    if echo_print == 1:
        print('\nExecute cmd instruction="{}"'.format(cmd_str))
    run(cmd_str, shell=True)
if __name__=='__main__':
    
    current_folder = os.path.dirname(os.path.abspath(__file__))

    
    os.chdir(current_folder)
    thresh = 0.01

    st_lib = 'data/Ray2013_rbp_Homo_sapiens.meme'
    temp_save_dir = './temp'

    with open('./temp_id_list.txt', 'r') as file:
        lines = file.readlines()

    # Remove line breaks at the end of each string
    motif_ids = [line.strip() for line in lines]
    dist = np.ones([len(motif_ids),len(motif_ids)])

    # Calculate which motifs are connected in sequence
    for ind, id in enumerate(motif_ids):
        print(id)
        run_cmd('meme-get-motif %s -id %s > %s.meme'%(st_lib,id,temp_save_dir))
        run_cmd('tomtom -thresh %s -norc -o %s %s.meme %s'%(thresh,temp_save_dir,temp_save_dir, st_lib))
        tom_tsv_dir = os.path.join(temp_save_dir,'tomtom.tsv')
        df = pd.read_csv(tom_tsv_dir, delimiter='\t')
        df_trimmed = df.iloc[:-3, :]
        assert(len(df_trimmed)>=1)

        if(len(df_trimmed)==1):
            dist[ind, ind] = 1
            #  Only match oneself, skip
            run_cmd('rm -r %s'%temp_save_dir)
            continue

        target_id = df_trimmed['Target_ID'].values
        qvalues = df_trimmed['q-value'].values
        for i in range(target_id.shape[0]):
            x_id = target_id[i]
            x_ind = motif_ids.index(x_id)
            dist[ind, x_ind] = qvalues[i]
            dist[x_ind, ind] = qvalues[i]
        run_cmd('rm -r %s'%temp_save_dir)
    joblib.dump(dist, '../data/dist.pkl')
    
        
        