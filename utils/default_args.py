

import argparse

class dotdict(dict):
    # https://stackoverflow.com/questions/42272335/how-to-make-a-class-which-has-getattr-properly-pickable

    __slots__ = ()

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __setattr__(self, key, value):
        self[key] = value

        
def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='configurations')
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--lr', type=float, default=1e-3, help='lr')
    parser.add_argument('--l2', type=float, default=1e-5, help='regularization')
    parser.add_argument('--dropout', type=float, default=0.1, help='drop')
    parser.add_argument('--batch_size', type=int, default=128, help='bs')
    parser.add_argument('--patience', type=int, default=100, help='patience')
    parser.add_argument('--epochs', default=1000, type=int, help='max_eps')
    parser.add_argument('--output_dir', required=False,default='results/train_wholeseq_multi_task',
                        help='path where to save, empty for no saving')
    parser.add_argument('--data_path', default='../data/f0_c0_data0_multi_task.h5', type=str,
                        help='dataset path')
    parser.add_argument('--checkpoint', default=None, type=str,
                help='')
    parser.add_argument('--description', default=None, type=str,
                    help='training description')
    

    args = parser.parse_args()
    args = dotdict(vars(args))
    return args


def get_explain_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config_file 在training阶段无用')
    parser.add_argument('--name', type=str, required=True, help='conv_layer_name')
    parser.add_argument('--ind', type=str, default='0-64', help='neuron_index')
    parser.add_argument('--gpu_id', type=str, default='0', help='which gpu to user')
    parser.add_argument('--batch_size', type=int, default=128, help='bs')
    parser.add_argument('--model_name',default="unkown_model", help='model_name,will be set as a workdir')
    parser.add_argument('--maxactivation_savedir', default='./max_activation',
                    help='path where to save, empty for no saving')
    parser.add_argument('--collected_samples_savedir', default='./maxactivation_samples',
                help='path where to save, empty for no saving')
    parser.add_argument('--all_PFM_save_dir', default='./ga_pfm',
            help='path where to save, empty for no saving')
    parser.add_argument('--cluster_save_dir', default='./clustering',
            help='path where to save, empty for no saving')
    parser.add_argument('--st_lib', default='../data/human_rbp_microrna_motif.meme',
            help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--checkpoint', default=None, type=str,
                        help='model_weights')
    parser.add_argument('--featuremap_layer_name_dict_path', default=None, type=str,
                    help='cluster criteria')
    parser.add_argument('--recp_path', default=None, type=str, help='receptive field')
    parser.add_argument('--data_path', default=None, type=str,
                    help='dataset path')
    parser.add_argument('--resample', default=False, type=bool,
                help='whether to resample')
    parser.add_argument('--recluster', default=False, type=bool,
            help='whether to re cluster')
    parser.add_argument('--ga_patience', default=30, type=int,
                help='sample patience')
    parser.add_argument('--thresh', type=float, default=0.01, help='qvalue thresh for tomtom')
    parser.add_argument('--alphabet', default='RNA', help='alphabet')

    args = parser.parse_args()
    args = dotdict(vars(args))
    return args
