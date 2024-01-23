import numpy
import pandas as pd
import os
import torch
import numpy as np
import random
import h5py
import seaborn as sns
from matplotlib import pyplot as plt
def one_hot_encode(df, col='utr', seq_len=50):
    # Dictionary returning one-hot encoding of nucleotides.
    nuc_d = {'a': [1, 0, 0, 0], 'c': [0, 1, 0, 0], 'g': [0, 0, 1, 0], 't': [0, 0, 0, 1], 'n': [0, 0, 0, 0]}

    # Creat empty matrix.
    vectors = np.empty([len(df), seq_len, 4])

    # Iterate through UTRs and one-hot encode
    for i, seq in enumerate(df[col].str[:seq_len]):
        seq = seq.lower()
        a = np.array([nuc_d[x] for x in seq])
        vectors[i] = a
    return vectors

if __name__ == '__main__':
    from scipy import stats
    
    current_folder = os.path.dirname(os.path.abspath(__file__))

    
    os.chdir(current_folder)

    save_dir = '../data/utr_mrl_non_AUG_alldataset.csv'
    utr_data_path = '../data/GSM3130435_egfp_unmod_1.csv'

    df = pd.read_csv(utr_data_path)
    df.sort_values('total_reads', inplace=True, ascending=False)
    df.reset_index(inplace=True, drop=True)
    df = df.iloc[:280000]
    df = df[~df['utr'].str.contains('ATG')]

    sns.kdeplot(df['rl']) 
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('mean:%.4f median:%.4f max:%.4f min:%.4f'%(df['rl'].mean(),df['rl'].median(),df['rl'].max(),df['rl'].min(),))
    
    plt.savefig('./image.png')
    df.to_csv(save_dir,index=False)


    
