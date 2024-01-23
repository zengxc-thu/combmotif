import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
def custom_sort(item):
    # If the string starts with 'hsa -', return a tuple, 
    # with the first element being 0 and the second element being the original string
    return (0, item) if item.startswith('hsa-') else (1, item)


if __name__ == '__main__':

    current_folder = os.path.dirname(os.path.abspath(__file__))

    os.chdir(current_folder)

    # To generate motif_scramble_stats.csv, plz follow the instruction of motif interaction
    df = pd.read_csv('../../motif_interaction/results/stats/hl_predictor/motif_scramble_stats.csv')

    motifA_list = df['motifA_id'].unique().tolist()
    motifB_list = df['motifB_id'].unique().tolist()
    motif_list = list(set(motifA_list + motifB_list))
    motif_list = [x for x in motif_list if x != 'Unknown']
    motif_list = sorted(motif_list, key=custom_sort)

    rows, cols = len(motif_list), len(motif_list)

    # Create a nxn string matrix with an initial value of "No combination"
    matrix = [["No-combination" for _ in range(cols)] for _ in range(rows)]

    value = np.ones([len(motif_list),len(motif_list)]) * np.inf

    for index, row in df.iterrows():
        if row['interaction'] == 'none':
            continue
        if row['motifA_id'] not in motif_list or row['motifB_id'] not in motif_list:
            continue

        motifa_ind = motif_list.index(row['motifA_id'])
        motifb_ind = motif_list.index(row['motifB_id'])
        if(row['interaction-p'] < value[motifa_ind,motifb_ind]):
            if(row['interaction'] == 'addictive' and matrix[motifa_ind][motifb_ind] != 'No-combination'):
                continue
            value[motifa_ind,motifb_ind] = row['interaction-p']
            value[motifa_ind,motifb_ind] = value[motifb_ind,motifa_ind]
            
            if('synergistic' in row['interaction']):
                matrix[motifa_ind][motifb_ind] = 'synergistic'
            elif('antagonistic' in row['interaction']):
                matrix[motifa_ind][motifb_ind] = 'antagonistic'
            else:
                matrix[motifa_ind][motifb_ind] = 'addictive'
            matrix[motifb_ind][motifa_ind] = matrix[motifa_ind][motifb_ind]

    colors = {'addictive': 'blue',
              'antagonistic': 'green', 
              'synergistic': 'orange', 
              'No-combination': 'white'}
    matrix = np.array(matrix)


    fig, ax = plt.subplots(figsize=(8, 8))


    # Traverse the matrix, draw points based on element values and add labels
    for label, color in colors.items():
        indices = np.argwhere(matrix == label)
        ax.scatter(indices[:, 1], indices[:, 0], c=color, marker='o', s=50, label=label)
    # Draw diagonal lines
    ax.plot([0, cols - 1], [0, rows - 1], color='lightgray', linestyle='--')
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels(motif_list, rotation=90)  
    ax.set_yticklabels(motif_list)

    plt.xlabel('Motif A')
    plt.ylabel('Motif B')
    plt.title('Motif Interaction Map')

    ax.grid(True)

    ax.legend(loc='upper right',bbox_to_anchor=(1.55, 0.7))
    plt.subplots_adjust(top=0.70, right=0.70, bottom=0.2, left=0.2)
    
    plt.savefig('motif_interaction_map.png')