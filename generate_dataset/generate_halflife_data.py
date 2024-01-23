import tensorflow as tf
import sys
from natsort import natsorted
import numpy as np
import glob
import h5py
from basenji import dataset



if __name__=='__main__':


  data_dir = 'data/f0_c0/data0' # human
  # data_dir = 'data/f0_c1/data0' # mouse
  dataset_split = ['train','test','valid']
  save_dir = 'data/%s_%s_wholeseq.h5'%(data_dir.split('/')[1],data_dir.split('/')[2])
  # with h5py.File(save_dir, 'w') as f:
  with h5py.File(save_dir, 'w') as f:
    for mode in dataset_split:

      my_dataset = dataset.RnaDataset(data_dir,
          split_label=mode,
          batch_size=32,
          mode=mode)
      seq, target = my_dataset.numpy()
      seq = np.squeeze(seq)
      target = np.squeeze(target)
      seq = np.transpose(seq,(0,2,1))
      print(seq.shape)
      print(target.shape)
      f.create_dataset('%s_label'%mode, data=target)
      f.create_dataset('%s_sequence'%mode, data=seq)


