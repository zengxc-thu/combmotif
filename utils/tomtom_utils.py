import numpy as np
import pandas as pd
max_length = 100

def divide_2_0(pfm, info_threshhold=20, adj_diff=0.5, valid_hold = 0.7, filter=False, min_base_num=4, valid_mean = False):


    pfm = pfm / pfm.sum(axis=0)
    zero_index = np.where(pfm==0)
    m = pfm.copy()
    m[zero_index[0], zero_index[1]]=1e-7
    H=-np.sum(np.log2(m)*pfm,axis=0)
    R=2-H
    pfm = pfm*R
    fitness = np.sum(pfm,axis=0)
    cumsum_fit = np.cumsum(fitness)
    p1 = 0
    p2 = 0
    motif_range = []
    while(1):
        if np.max(cumsum_fit) <= info_threshhold + cumsum_fit[p1]:
            while (1):
                p2 = p1 + max_length - 1
                motif_range.append([p1, p2])
                p1 = p2 + 1
                if cumsum_fit.shape[0] - p1 + 1 < max_length:
                    motif_range.append([p1, cumsum_fit.shape[0] - 1])
                    break
            break

        ind = np.where(cumsum_fit > cumsum_fit[p1] + info_threshhold)

        try:
            p2 = ind[0][0] - 1
        except:
            print('e')

        try:
            while (1):
                if p2 == cumsum_fit.shape[0]-1:
                    break
                if ind[0][0]>cumsum_fit.shape[0]-1 or cumsum_fit[p2+1]-adj_diff<=cumsum_fit[p2]:
                    break
                p2 += 1

        except:
            print('ee')

        if p2 - p1 >= max_length:
            p2 = p1 + max_length - 1
   
        motif_range.append([p1,p2])

        p1 = p2 + 1
        if p1 == cumsum_fit.shape[0]:
            break

        try:

            if cumsum_fit[p1] + info_threshhold >= cumsum_fit[-1] :
                if cumsum_fit.shape[0] - p1 + 1 <= max_length:
                    motif_range.append([p1, cumsum_fit.shape[0]-1])
                    break
                else:
                    while(1):
                        p2 = p1 + max_length -1
                        motif_range.append([p1, p2])
                        p1 = p2 + 1
                        if cumsum_fit.shape[0] - p1 + 1  < max_length:
                            motif_range.append([p1, cumsum_fit.shape[0]-1])
                            break

                    break
        except:
            print('ee')

    refined_motif_range = []
    base_num = 3
    upper_hold = 30
    lower_hold = min_base_num
    if valid_mean:
        valid_hold = fitness.mean()

    for block_range in motif_range:
        start, end = block_range[0], block_range[1]
        valid_flag = 0
        for p in np.arange(start, end - (base_num - 2)):
            # print(p)
            if np.mean(fitness[p : p + base_num]) < valid_hold and not valid_flag:
                continue
            if np.mean(fitness[p : p + base_num]) > valid_hold and not valid_flag:
                p1 = p
                valid_flag = 1
            elif np.mean(fitness[p : p + base_num]) < valid_hold and valid_flag:
                p2 = p + base_num - 1
                valid_flag = 0
                if p2 - p1 < upper_hold and p2 - p1 > lower_hold :
                    refined_motif_range.append([p1, p2])
            elif p == end - 2 and valid_flag:
                p2 = p
                if p2 - p1 < upper_hold and p2 - p1 > lower_hold :
                    refined_motif_range.append([p1, p2])


    fitered_motif_range = []
    if filter:
        for ind, block_range in enumerate(refined_motif_range):
            start, end = block_range[0], block_range[1]
            max_ind = pfm[:,start:end+1].argmax(axis=0)
            max_stats = pd.DataFrame(max_ind).value_counts()

            if max_stats.iloc[0]/max_ind.shape[0]>0.8:
                # print(max_ind)
                continue
            elif (max_stats.iloc[0]+max_stats.iloc[1])/max_ind.shape[0]> 0.9:
                # print(max_ind)
                continue

            fitered_motif_range.append(block_range)
    else:
        fitered_motif_range = refined_motif_range


    return fitered_motif_range


def convert_dna_to_rna_meme(input_file, output_file):
    with open(input_file, 'r') as infile:
        meme_content = infile.read()

    # replace 'T'  of 'U'
    meme_content = meme_content.replace('T 0.25000', 'U 0.25000')

    # update ALPHABET to RNA
    meme_content = meme_content.replace('ALPHABET= ACGT', 'ALPHABET= ACGU')

    with open(output_file, 'w') as outfile:
        outfile.write(meme_content)
