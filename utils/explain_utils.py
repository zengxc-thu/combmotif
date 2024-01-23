import numpy as np
import torch
from .train_utils import get_eval_dataloader
import re
import random
import utils

def cluster_name_to_file_path(base_path, cluster_name):
    pattern = re.compile(r'\d+')

    res = re.findall(pattern, cluster_name)

    file_path = base_path
    for i in res:
        if i!='0' and i.isdigit():
            file_path = file_path + '/' + i
    return file_path

def record_pfm_id_info(cluster_name, pfm, width, height, head, body, max_activation=None):

    PPM = pfm / pfm.sum(axis=0)
    zero_index = np.where(PPM==0)
    m = PPM.copy()
    m[zero_index[0], zero_index[1]]=1e-7
    if PPM.shape[0]==4:
       H=-np.sum(np.log2(m)*PPM,axis=0)
    else:
       raise Exception("PPM dim error!!")
    R=2-H
    PWM = PPM*R


    PWM = {"A": list(PWM[0, :]), "C": list(PWM[1, :]), "G": list(PWM[2, :]), "U": list(PWM[3, :])}
    if max_activation==None:
        head += '<tr><td>%s</td><td><canvas id="%s"></canvas></td></tr>\n' % (cluster_name, cluster_name)
    else:
        head += '<tr><td>%s_max%.4f</td><td><canvas id="%s"></canvas></td></tr>\n' % (cluster_name, max_activation, cluster_name)
    body += 'var data =%s;sequence_logo(document.getElementById("%s"), %s,%s, data, options)\n' % (PWM, cluster_name,width, height)

    return head,body
def record_pfm_id_info_for_saliency(cluster_name, PWM, width, height, head, body,  max_activation=None):
    PWM = {"A": list(PWM[0, :]), "C": list(PWM[1, :]), "G": list(PWM[2, :]), "T": list(PWM[3, :])}
    if max_activation==None:
        head += '<tr><td>%s</td><td><canvas id="%s"></canvas></td></tr>\n' % (cluster_name, cluster_name)
    else:
        head += '<tr><td>%s_max%.4f</td><td><canvas id="%s"></canvas></td></tr>\n' % (cluster_name, max_activation, cluster_name)
    body += 'var data =%s;sequence_logo(document.getElementById("%s"), %s,%s, data, options)\n' % (PWM, cluster_name,width, height)

    return head,body
def get_fitness_2_0(pop,network,device,neuron_index,best_one,max_activation, counter,featuremap_layer_name, is_top_layer=True):
    pop = pop.to(device)
    with torch.no_grad():
        pred = network(pop, featuremap_layer_name, is_top_layer)
        pred = pred[:,neuron_index,:]
        if torch.max(pred)>max_activation:
            max_activation = torch.max(pred)
            best_one = pop[torch.argmax(pred),...]
            print('\n')
            counter = 0
        print('max_activation:%s   counter:%s   ' %(torch.max(pred), counter))
        counter = counter + 1
        return (pred - torch.min(
            pred)) + 1e-3,best_one, max_activation, counter  
def get_fitness_and_activation_2_0(pop,network,device,neuron_index, featuremap_layer_name, args_batch_size=256, is_top_layer=True):

    dataloader = get_eval_dataloader(pop, args_batch_size)
    pred = torch.zeros([pop.shape[0],1])
    pred = pred.to(device)
    network.eval()
    cnt = 0
    with torch.no_grad():
        for step, (batch_x)  in enumerate(dataloader):  
            batch_x = batch_x[0].to(device)
            output = network(batch_x, featuremap_layer_name, is_top_layer)
            pred[cnt:cnt+output.shape[0],...] = output[:, neuron_index, :]
            cnt += output.shape[0]
        return (pred - torch.min(
            pred)) + 1e-7,pred 
def get_featuremap_for_cluster(network, featuremap_layer_name, samples_of_cluster, is_top_layer = False, args_batch_size=256):
    """
    samples_of_cluster:tensor num*4*sense_field
    featuremap_layer_name: the featmap for cluster
    test_output:numpy array num*....
    """
    network.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if featuremap_layer_name != 'input':
        with torch.no_grad():
            dataloader = get_eval_dataloader(samples_of_cluster, args_batch_size)
            cnt = 0
            for step, (batch_x) in enumerate(dataloader):  
                batch_x = batch_x[0].to(device)
                output = network(batch_x, featuremap_layer_name, is_top_layer)
                if step == 0:
                    test_output = torch.zeros([samples_of_cluster.shape[0], output.shape[1], output.shape[2]])
                    test_output = test_output.to(device)
                test_output[cnt:cnt + output.shape[0], ...] = output
                cnt += output.shape[0]
    else:
        test_output = samples_of_cluster

    test_output = test_output.cpu().numpy()
    return test_output
def get_activation(pop,network,conv_layer_name,device):
    dataloader = get_eval_dataloader(pop, args_batch_size=256)
    network.eval()
    cnt = 0
    with torch.no_grad():
        for step, (batch_x)  in enumerate(dataloader):  
            batch_x = batch_x[0].to(device)
            output = network(batch_x,conv_layer_name)

            if step==0:
                pred = torch.zeros([pop.shape[0], output.shape[1]])
                pred = pred.to(device)
            pred[cnt:cnt+output.shape[0],...] = output[:, :, 0]
            cnt += output.shape[0]
    return pred

def resample_from_trainingset(train_sequence, trian_label, n, length):
    seq_cnt = 0
    xs = np.zeros([n,train_sequence.shape[1],length])

    while seq_cnt<n:

        random_seq_id = random.randint(0, train_sequence.shape[0]-1)
        if(train_sequence.shape[2]>=length):
            random_start_point = random.randint(0, train_sequence.shape[2] - length)
            xs[seq_cnt,...] = train_sequence[random_seq_id,:,random_start_point:random_start_point+length]
        else:
            random_start_point = random.randint(0, length - train_sequence.shape[2])
            xs[seq_cnt,:,random_start_point:random_start_point+train_sequence.shape[2]] = train_sequence[random_seq_id,...]

        seq_cnt+=1

    
    return xs

            