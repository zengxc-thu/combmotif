import h5py
import os
import numpy as np
import sys

sys.path.append("./")
import utils

hdf5_results = h5py.File("motif_discovery/tf_modisco/modisco_results/hl_predictor/conv7_neuron4_tfmodisco.h5" , "r")


motif_pattern = {}
for activity in list(hdf5_results.keys()):
    for pattern in list(hdf5_results[activity].keys()):
        motif_pattern[f'{activity}_{pattern}'] = hdf5_results[activity][pattern]['sequence'][:]
print(motif_pattern)

save_path = './'
match_flag = False
if not os.path.exists('tomtom_test/tomtom.tsv'):
    for pattern_name in motif_pattern.keys():
        print(pattern_name)
        info = motif_pattern[pattern_name]
        pwm = info / info.sum(axis=1).reshape(info.sum(axis=1).shape[0], 1)
        back = np.zeros_like(pwm)
        pfm = np.floor(pwm * 1000)
        back[:, 0] = 1000 - np.floor(pwm * 1000).sum(axis=1)
        pfm = pfm + back

        info = np.transpose(info)
        pfm = np.transpose(pfm)

        motif_range = utils.divide_2_0(info.copy())
    
        for j, ind in enumerate(motif_range):
            match_flag = True


            motif_i = pfm[:, ind[0]: ind[1] + 1]
            cluster_name = f'{pattern_name}.{j}-{ind[0]}to{ind[1]}' 
            print(cluster_name)

            f = open('%s/convx.chen' % (save_path), 'a')

            if motif_i.shape[0] == 4:
                motif_i = np.transpose(motif_i)
            for index, c in enumerate(motif_i):
                if index == 0:
                    f.writelines('>%s\n' % cluster_name)
                f.writelines('%d %d %d %d\n' % (c[0], c[1], c[2], c[3]))
            f.close()


hdf5_results.close()

# if(match_flag):
#     run_cmd('chen2meme %s/%s.chen > %s/%s.meme' % (save_path, task, save_path, task))
#     if(args.alphabet=='RNA'):
#         utils.convert_dna_to_rna_meme('%s/%s.meme' % (save_path, task), '%s/%s.meme' % (save_path, task))
#     run_cmd('tomtom -thresh %s -norc -o %s/tomtom_out_%s %s/%s.meme %s' % (thresh, save_path, task, save_path, task, args.st_lib))
#     run_cmd('rm %s/%s.chen' % (save_path, task))
#     run_cmd('rm %s/%s.meme' % (save_path, task))
# else:
#     print("No query motif for %s. Current neuron doesn't have any preferred patterns" %task)
#     continue
