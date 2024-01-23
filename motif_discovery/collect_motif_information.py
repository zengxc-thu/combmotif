import os
import sys
sys.path.append('/home/xczeng/postgraduate/saluki_torch-master')
import utils
import pandas as pd
import joblib
import numpy as np
import re
def get_newest_folder(directory):
    # Retrieve the paths and creation times of all folders in the directory and store them in a list.
    folders = [(os.path.join(directory, folder), os.path.getctime(os.path.join(directory, folder)))
               for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

    # If there are no folders in the directory, return None.
    if not folders:
        return None

    # Sort the folders by creation time, with the newest folder placed at the end of the list.
    newest_folder = sorted(folders, key=lambda x: x[1])[-1][0]

    return newest_folder

if __name__=='__main__':
    
    current_folder = os.path.dirname(os.path.abspath(__file__))

    
    os.chdir(current_folder)

    args = utils.get_explain_args()

    prefix = './tomtom_match_results/%s/' % args.model_name 

    conv_list = os.listdir(prefix)
    ind = re.findall("\d+", args.ind)
    neuron_list = np.arange(int(ind[0]),int(ind[1]))

    thresh = 0.01

    df_all = pd.DataFrame()

    save_dir = os.path.join('motif_information', args.model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for conv_ind in conv_list:
        conv_dir = prefix + conv_ind
        conv_dir = get_newest_folder(conv_dir) #Get the latest version in the "conv_dir" directory.
        for neuron_id in neuron_list:
            task = '%s_neuron%d'%(conv_ind, neuron_id+1)
            neuron_dir = "%s/tomtom_out_%s/tomtom.tsv"%(conv_dir, task)

            if(not os.path.exists(neuron_dir)):
                print('-----------%s:no validly matched motifs--------------'%task)
                continue
            else:
                df = pd.read_csv(neuron_dir, sep='\t', header=0, comment='#')
                if(len(df)==0):
                    print('-----------%s:no validly matched motifs--------------'%task)
                    continue
            print('-----------%s:find matched motifs--------------'%task)  
            df['task'] = task
            df['path'] = neuron_dir
            df_all = pd.concat([df_all, df], axis=0)

    if(len(df_all) == 0): 
        print(f"There are no matched motifs in {task}")
        sys.exit()

    df_all = df_all[df_all['q-value']<1e-2]
    df_all = df_all.reset_index()  #Reset the index to facilitate subsequent filtering operations.

    print('\n-----------The following neurons have successfully matched motifs q-value,thresh%.4f-----------\n'%thresh)
    print(df_all['task'].unique())
    joblib.dump(df_all['task'].unique(),f'{save_dir}/matched_neurons.pkl')

    print('\n-----------A total of %d motifs were matched. q-value,thresh%.4f-----------\n'%(df_all['Target_ID'].nunique(),thresh))

    # Many motifs from the RBP database are matched successfully. Selected the sample with the lowest q-value for each matched motif.
    significant_motif = df_all.loc[df_all.groupby('Target_ID')['q-value'].idxmin()]


    # Drop duplicates. If a motif is recognized by only one neuron, select the neuron that recognizes it the best. Delete the records of other neurons that match the same motif.
    df_x = df_all.copy()

    for index, row_y in significant_motif.iterrows():
        first_sample_A = row_y['Target_ID']
        first_sample_B = row_y['task']
        df_x = df_x[~((df_x['Target_ID'] == first_sample_A) & (df_x['task'] != first_sample_B))]

    df_x = df_x.reset_index(drop=True)

    print('\n-----------After deduplication, a total of %d neurons matched unique motifs.-----------\n'%len(df_x['task'].unique()))




    # Print the top five matched motifs with the lowest q-values.
    print('\n-----------The top five motifs with the best matches.-----------\n')
    print(significant_motif.sort_values(by='q-value').head()[['Target_ID','q-value','task','Query_ID','path']])




    # Each neuron primarily recognizes which motifs, meaning, for each neuron, identify the motifs it matches and select the query with the minimum q-value for each motif.
    neuron_motif = df_all.loc[df_all.groupby(['task', 'Target_ID'])['q-value'].idxmin()]
    neuron_motif = neuron_motif.sort_index()

    print('\n-----------The primary motif recognized by each neuron.-----------\n')
    tmp1 = neuron_motif.loc[neuron_motif.groupby(['task'])['q-value'].idxmin()].copy()
    tmp1 = tmp1.sort_index()
    print(tmp1[['task','Target_ID','Query_ID','q-value','path']])

    print('\n-----------Which motifs are matched the most?-----------\n')
    # Each occurrence of a motif recognized by a particular neuron (with sample q-value < threshold) is counted as one occurrence.
    print(neuron_motif['Target_ID'].value_counts().sort_values(ascending=False))


    # which neurons recognize each motif.
    print('\n-----------which neurons recognize each motif.-----------\n')
    tmp2 = neuron_motif.groupby('Target_ID')['task'].apply(set)
    print(tmp2.sort_values(ascending=False, key=lambda x: x.apply(len)))# 按照每个motif识别神经元的数量由高到低排列一下 



    with pd.ExcelWriter(f'{save_dir}/output.xlsx') as writer:

        significant_motif[['Target_ID','Query_ID','q-value','task','path']].to_excel(writer, sheet_name='samples that represents the lowest q-value match.', index=False)
        significant_motif.sort_values(by='q-value').head()[['Target_ID','q-value','task','Query_ID','path']].to_excel(writer, sheet_name='The top five motifs with the best matches (lowest q-values).', index=False)
        tmp1[['task','Target_ID','Query_ID','q-value','path']].to_excel(writer, sheet_name='Each neuron primarily prefers which motif.', index=False)
        neuron_motif.sort_index()[['task','Target_ID','Query_ID','q-value','path']].to_excel(writer, sheet_name='Each neuron matches with which motifs.', index=False)
        neuron_motif['Target_ID'].value_counts().sort_values(ascending=False).to_excel(writer, sheet_name='Which motifs are matched the most?', index=True)
        tmp2.sort_values(ascending=False, key=lambda x: x.apply(len)).to_excel(writer, sheet_name='Which neurons recognize each motif individually.', index=True)



    









