# -*- coding = utf-8 -*-
# @Time : 2024/1/15 14:33
# @Author : zxc
# @File : neuronMotif_adaptive_sample_cluster.py
# @Software : Genetic algorithm sampling combined with backward layer-wise clustering.
import torch
import numpy as np
import sys
from model_zoo import Saluki_Motif
from utils import (get_region, generate_initial_group,
    get_gredient_2_0, probability_transform_3_0, mutation_13_0,
    cyclic_shift_2_0, cross_over_5_0, reinitial_same_pop, seq_to_name, 
    get_fitness_and_activation_2_0,get_featuremap_for_cluster,cluster_name_to_file_path,
    record_pfm_id_info)
import joblib
import utils
import pandas as pd
from sklearn.cluster import KMeans
import os
import time
import datetime
import random
import re


def run_cmd(cmd_str='', echo_print=0):

    from subprocess import run
    if echo_print == 1:
        print('\nExecute cmd instruction="{}"'.format(cmd_str))
    run(cmd_str, shell=True)


def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


"""
For a population of 50 samples, in addition to updating each sample using the genetic algorithm, 
gradient updates are also applied to each sample. Perform one forward pass for each sample, 
then obtain the gradients of the function with respect to each parameter of each sample.
"""
if __name__ == "__main__":
    
    current_folder = os.path.dirname(os.path.abspath(__file__))

    
    os.chdir(current_folder)

    args = utils.get_explain_args()

    cfgs = utils.read_yaml_to_dict(args.config)
    
    for key in cfgs.keys():
        if(key == 'ind' and cfgs['multi_thread']):
            continue
        args[key] = cfgs[key]
        


    conv_layer_name = args.name
    ind = re.findall("\d+", args.ind)
    neuron_index_list = np.arange(int(ind[0]), int(ind[1]))
    print(f'\nInterpreting model: {args.checkpoint}')
    print(f'NeuronMotif info: Sample and cluster on : {conv_layer_name}_neuron{neuron_index_list[0] + 1} ... {conv_layer_name}_neuron{neuron_index_list[-1] + 1}\n')

    set_random_seed(args.seed)
    dt = datetime.datetime.now()

    if("conv" in args and args.conv != None):
        m = re.findall("\d+", args.conv)
        args.conv = [int(i) for i in m]
    if("bl" in args and args.bl != None):
        m = re.findall("\d+", args.bl)
        args.bl = [int(i) for i in m]
    if("fc" in args and args.fc != None):
        m = re.findall("\d+", args.fc)
        args.fc = [int(i) for i in m]

    sense_field = utils.read_yaml_to_dict(args.recp_path)

    cluster_repeat = args.resample
    sample_repeat = args.recluster

    all_PFM_save_path = "%s/%s"%(args.all_PFM_save_dir,args.model_name) ## The samples obtained through clustering the acquired samples.
    sample_save_path = "%s/%s"%(args.collected_samples_savedir,args.model_name)  ## The save dir for the collected samples.

    utils.create_dir(all_PFM_save_path)
    utils.create_dir(sample_save_path)


    region_nums = 20
    sample_num = int((sense_field[conv_layer_name] / 100 * 2000) // 100 * 100 + 100)
    # sample_num = 100
    max_time = 1200
    max_iter = 350
    patience = args.ga_patience #30
    min_time = 0  

    POP_SIZE = 1000
    cyclic_shift_pop_size = int(POP_SIZE * 0.2)
    top_fitness_pop_size = int(POP_SIZE * 0.4) 

    random_pop_size = POP_SIZE - cyclic_shift_pop_size - top_fitness_pop_size
    start_max_MUTATION_RATE = 0.4 
    min_MUTATION_RATE = 0.4 


    gredient_mutation_ratio = 0.2  #   If you need to further increase diversity, you can increase this ratio to 0.3, and you can also increase the mutation size.
    random_mutation_ratio = 0.6 #

    max_shift = 10000
    N_GENERATIONS = 50000000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cluster_network = Saluki_Motif(num_layers=args.num_layer,
                                   seq_depth = args.seq_depth, 
                                   num_targets = args.num_targets)

    cluster_network.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model_state_dict'])
    cluster_network.to(device)
    cluster_network.eval()
    
    # Read the clustering criteria.
    featuremap_layer_name_dict = utils.read_yaml_to_dict(args.featuremap_layer_name_dict_path)
    featuremap_layer_name_dict = featuremap_layer_name_dict[conv_layer_name]
    # print('loading cluster criteria from %s'%args.featuremap_layer_name_dict_path)

    ## get neuron nums
    tmp_pop = generate_initial_group(2, total_length=sense_field[conv_layer_name])
    tmp_output = cluster_network(tmp_pop.to(device), conv_layer_name, True)
    neuron_num = tmp_output.shape[1]

    history = []

    for neuron_index in neuron_index_list:
        max_act = -1000
        counter = 0
        MUTATION_RATE = start_max_MUTATION_RATE

        dict_of_labelm = {}
        neuronname = 'neuron' + str(neuron_index + 1)
        task = conv_layer_name + '_' + neuronname
        if(neuron_index >= neuron_num):
            print(f"{task} doesn't exist , because {conv_layer_name} has {neuron_num} filters in total")
            continue
        print('*********************%s:START sampling************************' % task)

        # Partitioning the activation value range.
        region = get_region(max_value=region_nums, region_num=region_nums)

        if (not os.path.exists(sample_save_path + '/' + conv_layer_name + '_neuron' + str(
                neuron_index + 1) + '.pkl') or sample_repeat):

            # Generate the initial population.
            pop = generate_initial_group(POP_SIZE, total_length=sense_field[conv_layer_name])
            print(pop.shape)

            # Initialize variables.
            sample = torch.zeros([region_nums, sample_num, 4, sense_field[conv_layer_name]]) ## 用来放收集到的样本
            nums_in_each_region = torch.zeros([region_nums]).int()
            overflow_flag = torch.zeros([region_nums])

            st = time.time()
            for iter_i in range(N_GENERATIONS):  

                #Store the final population for this generation.
                final_pop = torch.zeros([POP_SIZE, 4, sense_field[conv_layer_name]])
                final_pop_cnt = 0 # Record the number of individuals already generated.

                # Obtain the gradient.
                gredient = get_gredient_2_0(pop.clone(), cluster_network, device, neuron_index, batch_cal=True, featuremap_layer_name=conv_layer_name)
                # Gradient -> Probability
                mutation_probability = probability_transform_3_0(gredient.cpu())

                # 1. Mutation = Gradient Mutation + Random Mutation
                pop = mutation_13_0(mutation_probability.clone(), pop.clone(), MUTATION_RATE,
                                    mutation_ratio=gredient_mutation_ratio)

                # 2.Evaluate the individual's fitness.
                fitness, activation_value = get_fitness_and_activation_2_0(pop.clone(), cluster_network, device,
                                                                       neuron_index, conv_layer_name)
                # 3.Natural selection.
                # Sort by fitness in descending order.
                fitness_descending = torch.sort(fitness, descending=True, dim=0)
                # 3.1 Select the top `top_fitness_pop_size` individuals with the highest fitness to directly enter the next generation.
                top_fitness_indices = fitness_descending.indices[0:top_fitness_pop_size]
                final_pop_cnt = top_fitness_indices[:, 0].shape[0]
                final_pop[0:final_pop_cnt, ...] = pop[top_fitness_indices[:, 0], ...].clone()
                # 3.2 Select `cyclic_shift_pop_size` individuals obtained by cyclically shifting the individuals from the top fitness indices and directly incorporate them into the next generation.
                cyclic_shift_pop = cyclic_shift_2_0(pop[top_fitness_indices[:, 0], ...].clone(), max_shift,
                                                    cyclic_shift_pop_size, shift_rate=1)
                final_pop_cnt += cyclic_shift_pop_size
                final_pop[top_fitness_indices[:, 0].shape[0]: final_pop_cnt, ...] = cyclic_shift_pop.clone()

                # 3.3 In the previous generation, individuals undergo crossover exchanges according to probabilities to generate the next generation of individuals.
                random_pop = cross_over_5_0(pop.clone(), fitness.clone(), random_pop_size)
                final_pop[final_pop_cnt:, ...] = random_pop.clone()
                # 3.4 Remove duplicates.
                pop = reinitial_same_pop(final_pop.clone())

                # 4. Reevaluate the activation values of individuals for sampling.
                new_fitness, new_activation_value = get_fitness_and_activation_2_0(pop.clone(), cluster_network, device,
                                                                               neuron_index, conv_layer_name)

                # 4.1 Control the mutation rate based on the sampling situation, 
                # aiming to achieve high diversity in the collected samples. 
                # Additionally, use a counter to determine whether the algorithm has 
                # converged; if it has, stop the iteration.
                if iter_i <= patience:
                    max_MUTATION_RATE = start_max_MUTATION_RATE
                else:
                    max_MUTATION_RATE = 0.9

                decay_rate = (min_MUTATION_RATE / max_MUTATION_RATE) ** (1 / (patience))
                if torch.max(new_activation_value).item() - max_act >= 1e-2:
                    max_act = torch.max(new_activation_value).item()
                    counter = 0
                    MUTATION_RATE = max_MUTATION_RATE
                    print('iter:%s  max_activation:%.4f counter:%d mutation_rate:%.2f max_mutation_rate:%.2f' % (
                    iter_i, max_act, counter, MUTATION_RATE, max_MUTATION_RATE))
                else:
                    counter += 1
                    MUTATION_RATE = MUTATION_RATE * decay_rate
                    if MUTATION_RATE < min_MUTATION_RATE:
                        MUTATION_RATE = min_MUTATION_RATE
                    print('iter:%s  max_activation:%.4f counter:%d mutation_rate:%.2f max_mutation_rate:%.2f' % (
                    iter_i, max_act, counter, MUTATION_RATE, max_MUTATION_RATE))
                    if counter > patience:
                        break
                # 4.2 update region
                region = get_region(max_value=max_act, region_num=region_nums)
                # 5.1 Sample the current population within specified intervals.
                for m in range(region_nums):
                    index_for_labelm = torch.where(
                        (new_activation_value >= region[m]) & (new_activation_value < region[m + 1]))
                    samples_of_labelm = pop[index_for_labelm[0], ...].clone().cpu()
                    nums_of_labelm = index_for_labelm[0].shape[0]
                    temp_dict_for_labelm = {}

                    # 5.1 For intervals that have been fully sampled, lose samples with a certain probability.
                    if overflow_flag[m] == 1 and nums_of_labelm >= 1:
                        cnt = 0
                        for xx in samples_of_labelm:
                            temp_dict_for_labelm = {}
                            name = seq_to_name(xx.clone())
                            temp_dict_for_labelm['%s' % name] = xx
                            for mm in temp_dict_for_labelm.keys() - dict_of_labelm.keys():
                                cnt += 1
                                ind_old = np.random.choice(sample_num, size=int(1), replace=True)
                                old_name = seq_to_name(sample[m, ind_old, ...][0])
                                assert(f'{old_name}' in dict_of_labelm)
                                del dict_of_labelm[f'{old_name}']
                                sample[m, ind_old, ...] = temp_dict_for_labelm[mm]
                                dict_of_labelm['%s' % mm] = 0
                                if cnt > (sample_num / patience * 2):
                                    break
                            if cnt >= (sample_num / patience * 2):
                                break

                    # 5.2 collect samples
                    if nums_of_labelm >= 1 and overflow_flag[m] == 0:
                        for xx in samples_of_labelm:
                            name = seq_to_name(xx.clone())
                            temp_dict_for_labelm['%s' % name] = xx
                        for mm in temp_dict_for_labelm.keys() - dict_of_labelm.keys():
                            sample[m, nums_in_each_region[m], ...] = temp_dict_for_labelm[mm]
                            nums_in_each_region[m] += 1
                            dict_of_labelm['%s' % mm] = 0
                            if nums_in_each_region[m] >= sample_num:
                                overflow_flag[m] = 1
                                break

                end = time.time()
                print(f"Sample num in each bin({region_nums}in total): {nums_in_each_region.tolist()}", '%d'%(end - st), 's')
                k = torch.where(nums_in_each_region < sample_num)
                if k[0].shape[0] == 0 and end - st > min_time:
                    break

                # Stop when the maximum number of generations is exceeded.
                if iter_i > max_iter:
                    break


            # After sampling, perform post-processing on the data, organize, trim, and save.
            for label, sample_from_same_region in enumerate(sample):
                save_dir = './PFM_plus' + '/' + conv_layer_name + '_neuron' + str(neuron_index + 1)
                utils.create_dir(save_dir)

                PFM = torch.sum(sample_from_same_region, dim=0)
                joblib.dump(PFM, save_dir + '/region' + str(label) + '_with' + str(
                    nums_in_each_region[label].item()) + 'samples' + '.pkl')




            sample = sample.reshape(-1, 4, sense_field[conv_layer_name])

            none_zero_index = np.array([])
            for index, i in enumerate(sample):
                if i.any():
                    none_zero_index = np.append(none_zero_index, index)

            sample = sample[none_zero_index, ...]
            joblib.dump(sample.cpu(), sample_save_path + '/' + conv_layer_name + '_neuron' + str(
                neuron_index + 1) + '.pkl')
        else:
            sample = joblib.load(sample_save_path + '/' + conv_layer_name + '_neuron' + str(
                neuron_index + 1) + '.pkl')
            print('*********************%s:already sampled ************************' % task)

        # detect_same_pop(sample)
        # continue

        print('*********************%s:START clustering************************' % task)

        if not os.path.exists('%s/%s_all_PFM.pkl' % (all_PFM_save_path, task)) or cluster_repeat :
            if sense_field[conv_layer_name] < 7:
                min_num = 5
            else:
                min_num = 20
            args_batch_size = 256
            example_num = 10000000
            only_save_final_cluster = True

            print('conv_layer:%s, neuron:%d' % (conv_layer_name, neuron_index + 1))

            feat_history = []
            max_activation_path = conv_layer_name + '_' + neuronname + '.pkl'

            input_sequence = sample
            print(input_sequence.shape)

            if input_sequence.shape[0] <= 5:
                print(f'there is no activated samples for {task}, this neuron may be redundant')
                continue

            sample_need_cluster = {}  ##The cluster dictionary contains samples, not feature maps. It is cleared after each round.
            temp_cluster_res = {}  ##Temporary storage for samples sorted into classes.
            sample_finish_cluster = {}  ##Store the sample classes that have terminated iteration.
            max_activation_of_cluster = {}
            file_path_of_cluster = {}

            final_cluster_name = []
            cluster_base_path = "%s/%s/%s-mechanic/%s"%(args.cluster_save_dir,args.model_name,conv_layer_name,neuronname) + '/{}_{:02d}_{:02d}_{:02d}_{:d}'.format(
                dt.date(), dt.hour, dt.minute, dt.second,
                dt.microsecond)
            html_save_path = "%s/%s/%s-mechanic"%(args.cluster_save_dir,args.model_name,conv_layer_name)





            first_cluster = '0'
            sample_need_cluster[first_cluster] = input_sequence.to(device)  #

            tree_level = 0
            for index, (featuremap_layer_name, cluster_num) in enumerate(featuremap_layer_name_dict.items()):

                temp_cluster_res = {}
                for cluster_name, samples in sample_need_cluster.items():

                    if '+' not in featuremap_layer_name:
                        test_output = get_featuremap_for_cluster(cluster_network, featuremap_layer_name,
                                                                 samples_of_cluster=samples)
                    else:
                        feat1 = featuremap_layer_name.split('+')[0]
                        feat2 = featuremap_layer_name.split('+')[1]
                        test_output1 = get_featuremap_for_cluster(cluster_network, feat1, samples_of_cluster=samples)
                        test_output2 = get_featuremap_for_cluster(cluster_network, feat2, samples_of_cluster=samples)
                        padding_size = test_output2.shape[2] - test_output1.shape[2]
                        padding_term = torch.zeros(
                            [test_output1.shape[0], test_output1.shape[1], padding_size])  # 在右边padding0
                        test_output1 = np.concatenate((test_output1, padding_term), axis=2)
                        test_output = np.concatenate((test_output1, test_output2), axis=1)
                    # except:
                    #     print('error')

                    clf = KMeans(n_clusters=cluster_num)
                    s = clf.fit(test_output.reshape((test_output.shape[0], -1)))
                    for i in range(cluster_num):
                        subcluster_name = cluster_name + '-' + str(i + 1)
                        sample_index = np.where(clf.labels_ == i)[0]
                        sample_of_cluster_i = samples[sample_index, ...]
                        try:
                            num_flag = sample_of_cluster_i.shape[0] <= min_num * cluster_num  
                            sample_finish_cluster[subcluster_name] = sample_of_cluster_i
                            ##Record the file locations to save for each class.
                            file_path_of_cluster[subcluster_name] = cluster_name_to_file_path(cluster_base_path,
                                                                                              subcluster_name)
                            
                            if num_flag:
                                final_cluster_name.append(subcluster_name)
                                continue
                            else:
                                if featuremap_layer_name == list(featuremap_layer_name_dict.keys())[-1]:
                                    final_cluster_name.append(subcluster_name)
                                temp_cluster_res[subcluster_name] = sample_of_cluster_i
                        except:
                            continue

                sample_need_cluster = temp_cluster_res
                feat_history.append(featuremap_layer_name)
                if sample_need_cluster == {}:
                    break
                tree_level += 1
                print(f"cluster tree in level{tree_level}:",list(temp_cluster_res.keys()))



            js_path = './js/jseqlogo.js '
            head = """

            <html lang="en">
            <style>

            tr td,th{

            border:1px solid black;

            }

            .mt{

             border-collapse:collapse;

            }

            </style>
                <body>

                <table class="mt"><tr><td align=center>Cluster Name</td><td align=center>CN motifs</td></tr>
            """

            body = """
                    </table>
                    <script src="%s"></script>
                    <script>
                        var options = {
                            "colors": jseqlogo.colors.nucleotides
                        };
                   """
            body = body % js_path
            tail = """
                    </script>
                </body>
            </html>
            """

            utils.create_dir(cluster_base_path)


            all_PFM = []
            all_PFM_max = []
            CNT = 0
            for index, (cluster_name, sample) in enumerate(sample_finish_cluster.items()):

                if only_save_final_cluster == True and cluster_name not in final_cluster_name:
                    continue

                if sample.shape[0] < min_num:
                    continue

                sample_output = get_featuremap_for_cluster(cluster_network, conv_layer_name, samples_of_cluster=sample,
                                                           is_top_layer=True)[:, neuron_index, :]

                sample_output = np.reshape(sample_output, [-1])
                sample_max = sample_output.max()

                sample = sample.cpu().numpy()
                cluster_sample_num = sample.shape[0]
                PFM = np.sum(sample, axis=0)
                all_PFM.append(PFM.tolist())
                all_PFM_max.append(sample_max)

                head, body = record_pfm_id_info(cluster_name='%s:%s' % (CNT, cluster_name), max_activation=sample_max,
                                                pfm=PFM, width=sample.shape[2] * 20,
                                                height=100, head=head, body=body)
                CC = PFM.astype(int)
                CC = np.transpose(CC)

                CNT += 1

            html_text = head + body + tail

            # with open(cluster_base_path + '/%s.html'%task, 'w') as f:
            #     f.write(html_text)
            with open(os.path.join(cluster_base_path , '%s.html'%task), 'w') as f:
                f.write(html_text)

            f = open(cluster_base_path + '/cluster_info.txt', 'a')
            f.writelines(feat_history)
            f.close()
            if all_PFM == []:
                print("The cluster has too few samples.")
            joblib.dump(np.array(all_PFM), cluster_base_path + '/all_PFM.pkl')
            joblib.dump(np.array(all_PFM_max), cluster_base_path + '/all_PFM_max.pkl')
            joblib.dump(np.array(all_PFM), '%s/%s_all_PFM.pkl' % (all_PFM_save_path, task))
        else:
            print('*********************%s:already clustered ************************' % task)

