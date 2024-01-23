# -*- encoding: utf-8 -*-
'''
Filename         :neuronmotif_cnn.py
Description      :Motif models
Time             :2023/12/30 11:01:07
Author           :***
Version          :1.0
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .cnn_gru import StochasticShift
from .cnn_gru import Permute_GRU
from .cnn_gru import Permute_LN
import re



class Saluki_Motif(nn.Module):
    def __init__(self, seq_length=12288, seq_depth=6, augment_shift=3, filters=64, kernel_size=5, l2_scale=0.001,
                 ln_epsilon=0.007, activation="relu", dropout=0.1, num_layers=6, go_backwards=True, rnn_type="gru", residual=False, bn_momentum=0.90, num_targets=1):
        super(Saluki_Motif, self).__init__()
        self.seq_length = seq_length
        self.seq_depth = seq_depth
        self.augment_shift = augment_shift
        self.filters = filters
        self.kernel_size = kernel_size
        self.l2_scale = l2_scale
        self.residual = residual
        self.ln_epsilon = ln_epsilon
        self.activation = activation
        self.dropout = dropout
        self.num_layers = num_layers
        self.go_backwards = go_backwards
        self.rnn_type = rnn_type
        self.bn_momentum = bn_momentum
        self.num_targets = num_targets



        # RNA convolution
        self.conv1 = nn.Conv1d(in_channels=seq_depth, out_channels=filters, kernel_size=kernel_size, padding=0,
                                bias=False)
        if self.augment_shift != 0:
            self.shift_layer = StochasticShift(shift_max=self.augment_shift, symmetric=False, pad=0)
        
        self.Pln1 = Permute_LN(shape=(filters,), eps=self.ln_epsilon)

        # middle convolutions
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        
        for _ in range(num_layers):
            # self.bn_layers.append(nn.BatchNorm1d(filters, momentum=bn_momentum))
            self.ln_layers.append(Permute_LN(shape=(filters,), eps=self.ln_epsilon))
            self.conv_layers.append(nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size,
                                              padding=0, bias=True))
        
        # aggregate sequence

        self.bn1 = nn.BatchNorm1d(filters, momentum=bn_momentum)
        self.bn2 = nn.BatchNorm1d(filters, momentum=bn_momentum)
        self.fc1 = nn.Linear(filters, filters)
        self.head = nn.Linear(filters, self.num_targets)

        self.adapt_pool = nn.AdaptiveAvgPool1d(1)

        self.rnn_layer = Permute_GRU if rnn_type == 'gru' else nn.LSTM
        self.rnn = self.rnn_layer(input_size= filters, hidden_size=self.filters,batch_first=True, go_backwards=self.go_backwards)
        


       
    def forward(self, x, name, is_top_layer=False):
        assert(x.shape[1]==4)
        zeros1 = torch.zeros(x.shape[0], self.seq_depth - x.shape[1], x.shape[2]).to(x.device)
        x = torch.cat((x, zeros1), dim=1)

        x = self.conv1(x)
        if name == 'conv1':
            if(is_top_layer):
                return x[:, :, 0].unsqueeze(2)
            else:
                return x
        
        numbers = re.findall(r'\d+', name)
        cnt = int(numbers[0])
        if('maxpool' in name):
            cnt = cnt + 1
        for i in range(cnt - 1):
            x = self.ln_layers[i](x)
            x = F.relu(x)
            x = self.conv_layers[i](x)
            if(i == cnt - 2 and 'conv' in name):
                x = x
            else:
                x = F.max_pool1d(x,kernel_size=2)
        if(is_top_layer):
            return x[:, :, 0].unsqueeze(2)
        else:
            return x

def cal_sense_field(num_layers=6,kernel_size=5):
    sense_field = {}
    conv_num = num_layers + 1
    sense_field['conv1'] = kernel_size
    name_list = ['conv1']
    for i in range(2,num_layers+2):
        name_list.append('conv%d'%i)
        name_list.append('maxpool%d'%(i-1))
    for ind,name in enumerate(name_list):
        sense_field[name] = 1
        while(ind >= 0):
            if('maxpool' in name_list[ind]):
                sense_field[name] *= 2
            elif('conv' in name_list[ind]):
                sense_field[name] += kernel_size - 1
            ind -= 1
    for name in sense_field:
        print("sense_field['%s'] = %d"%(name,sense_field[name]))
    return sense_field






if __name__ == '__main__':
    import sys
    # sys.path.append('/mnt/disk1/xzeng/postgraduate/saluki_torch')
    # import utils
    # sense_field = cal_sense_field()

    # utils.save_dict_to_yaml(sense_field,'optimization/recp_field/resnet_1111.yaml')

    # print(sense_field)
    
    sense_field = {}
    sense_field['conv1'] = 5
    sense_field['conv2'] = 9
    sense_field['maxpool1'] = 10
    sense_field['conv3'] = 18
    sense_field['maxpool2'] = 20
    sense_field['conv4'] = 36
    sense_field['maxpool3'] = 40
    sense_field['conv5'] = 72
    sense_field['maxpool4'] = 80
    sense_field['conv6'] = 144
    sense_field['maxpool5'] = 160
    sense_field['conv7'] = 288
    sense_field['maxpool6'] = 320

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Saluki_Motif()
    # checkpoint = torch.load(
    #     'training_utr_detector/stats/resnet_utr_cds_metalen200_thresh50/utr_cds_dataset_metalen200_thresh50-resnet1_0_conv[2,3].pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.collect_blocks()
    model.to(device)

    # for i in range(12):
    # conv_layer_name = 'maxpool1'
    # print(conv_layer_name)


    n_sample = 2
    for conv_layer_name in sense_field:
        print("sense_field['%s'] = %d"%(conv_layer_name,sense_field[conv_layer_name]))
        test_input = torch.randn(
            n_sample, 4, sense_field[conv_layer_name], requires_grad=True)
        test_input = test_input.to(device)

        test_output = model(test_input, conv_layer_name)

        print(test_output.shape)


