# -*- coding = utf-8 -*-
# @Time : 2023/1/15 15:52
# @Author : zxc
# @File : ga_utils.py
# @Software : PyCharm

import torch
import numpy as np
from .train_utils import get_eval_dataloader


def get_region(max_value,region_num,min_value=0):
    return torch.linspace(min_value,max_value,region_num+1)


def generate_initial_group(num = 100, total_length = 10000):
    pop = np.zeros((num,4,total_length))
    for i in range(num):
        x = np.zeros([4, total_length])
        boundery = np.random.rand(1,total_length)
        index_A = np.where(boundery<0.25)
        index_C = np.where((boundery<0.5) & (boundery>=0.25))
        index_G = np.where((boundery<0.75) & (boundery>=0.5))
        index_T = np.where((boundery<1) & (boundery>=0.75))

        x[0,index_A[1]] = 1
        x[1,index_C[1]] = 1
        x[2,index_G[1]] = 1
        x[3,index_T[1]] = 1

        pop[i,...] = x
    pop = torch.from_numpy(pop).float()
    return pop

def get_gredient_2_0(pop,network,device,neuron_index, featuremap_layer_name,batch_cal=False, args_batch_size=256):
    if not batch_cal:
        pop = pop.to(device)
        pop.requires_grad_()
        output = network(pop ,featuremap_layer_name, True)
        m = torch.zeros_like(output)
        m [:,neuron_index,:] = 1  
        output.backward(m.float(),retain_graph=True)
        return pop.grad.data
    else:
        dataloader = get_eval_dataloader(pop, args_batch_size)
        gredient = torch.zeros_like(pop).to(device)
        cnt = 0
        for step, (batch_x) in enumerate(dataloader):  
            batch_x = batch_x[0].to(device)
            batch_x.requires_grad_()
            output = network(batch_x, featuremap_layer_name, True)
            m = torch.zeros_like(output)
            m[:, neuron_index, :] = 1  
            output.backward(m.float(), retain_graph=True)
            gredient[cnt:cnt + output.shape[0], ...] = batch_x.grad.data
            cnt += output.shape[0]
        return gredient



def probability_transform_3_0(gradient):
    neg_ind = torch.where(gradient<=0)
    gradient[neg_ind[0],neg_ind[1],neg_ind[2]] = 1e-7
    s = torch.sum(gradient, dim=1)
    s = s.unsqueeze(1)
    mutation_pobability = gradient/s

    if np.isnan(torch.max(mutation_pobability)):
        print(np.where(np.isnan(mutation_pobability)==True))
        print(np.where(np.isnan(s) == True))
        print(np.where(np.isnan(gradient) == True))
        index = np.where(np.isnan(mutation_pobability)==True)
        print(gradient[index[0][0],:,index[2][0]])
        print(gradient[index[0][0], :, index[2][0]])
        print(s[index[0][0], :, index[2][0]])
    return mutation_pobability
def cal_mutation_point_based_on_info(mutation_size, p):
    #Calculate the information entropy of a probability matrix and provide suitable sites for change

    m = p.clone()
    zero_index = np.where(m <= 1e-7)
    m[zero_index[0], zero_index[1]] = 1e-7
    H = -torch.sum(torch.log2(m) * p, dim=0)
    R = 2 - H
    p = p * R
    fitness = torch.sum(p,dim=0)
    zero_index = torch.where(fitness <= 0)
    fitness[zero_index[0]] = 1e-7
    fitness = fitness.detach().cpu().numpy().ravel()
    try:
        idx = np.random.choice(np.arange(p.shape[1]), size=mutation_size, replace=True,
                           p=(fitness) / (fitness.sum()))
    except:
        print("m:",m)
        print("p:",p)
        print('error')

    return idx

def mutation_13_0(probability_matrix, pop, MUTATION_RATE=0.03, mutation_ratio = 0.05):
    for i, p in enumerate(probability_matrix):
        x = np.random.rand()
        if x < MUTATION_RATE/2:
            child = pop[i,...].clone()
            mutation_size = int(pop.shape[2]*mutation_ratio)
            # mutation_point = np.random.choice(np.arange(0, child.shape[1]), size=mutation_size, replace=False)
            # pm = p[:,mutation_point]
            # cal_info_mat(pm.clone(),mutation_point)
            mutation_point = cal_mutation_point_based_on_info(mutation_size, p.clone())
            pm = p[:, mutation_point]
            m = torch.zeros_like(pm)
            rndPoint = torch.rand(1, m.shape[1])
            flag = torch.zeros_like(rndPoint)
            accumulator = torch.zeros_like(rndPoint)
            for ind, val in enumerate(pm):
                accumulator += val
                index = torch.where(accumulator >= rndPoint)
                m[ind, index[1]] = 1
                cc = torch.where(flag == 1)
                m[ind, cc[1]] = 0
                flag[0, index[1]] = 1
            child[:,mutation_point] = m
            pop[i, ...] = child
        elif x < MUTATION_RATE:
            child = pop[i, ...].clone()
            mutation_size = int(pop.shape[2] * mutation_ratio)
            pm = torch.ones([4, mutation_size]) * 0.25
            mutation_point = np.random.choice(np.arange(0, child.shape[1]), size=mutation_size, replace=False)
            m = torch.zeros_like(pm)
            rndPoint = torch.rand(1, m.shape[1])
            flag = torch.zeros_like(rndPoint)
            accumulator = torch.zeros_like(rndPoint)
            for ind, val in enumerate(pm):
                accumulator += val
                index = torch.where(accumulator >= rndPoint)
                m[ind, index[1]] = 1
                cc = torch.where(flag == 1)
                m[ind, cc[1]] = 0
                flag[0, index[1]] = 1
            child[:, mutation_point] = m

            pop[i, ...] = child

    return pop
def cyclic_shift_2_0(pop,max_shift,cyclic_shift_pop_size, shift_rate = 0.2):
    new_pop = torch.zeros(cyclic_shift_pop_size, 4, pop.shape[2])
    cyclic_shift_cnt = 0
    while(1):
        for index, father in enumerate(pop):
            child = torch.tensor([])
            if np.random.rand() < shift_rate:
                cyclic_shift_cnt += 1
                length = father.shape[1]
                if max_shift>(length-1)/2:
                    max_shift = np.ceil((length - 1)/2)
                if np.random.rand() < 0.5:
                    #left move
                    K=np.random.choice(np.arange(1,max_shift+1), size=1, replace=True,
                           p=np.flipud(np.arange(1,max_shift+1)/np.arange(1,max_shift+1).sum()))
                    father = move(father, int(K))
                else:
                    #right move
                    K=np.random.choice(np.arange(1,max_shift+1), size=1, replace=True,
                           p=np.flipud(np.arange(1,max_shift+1)/np.arange(1,max_shift+1).sum()))
                    father = move(father, int(length-K))
                child = father
            if child.shape[0]>0:
                new_pop[cyclic_shift_cnt-1,...] = child.unsqueeze(0)
            if cyclic_shift_cnt == cyclic_shift_pop_size:
                break
        if cyclic_shift_cnt == cyclic_shift_pop_size:
            break

    return new_pop
def move(lists, K):  
    return torch.cat((lists[:, K % lists.shape[1]:], lists[:, :K % lists.shape[1]]), dim=1)
def cross_over_5_0(pop, fitness, child_size):

    new_pop = torch.zeros([child_size, pop.shape[1], pop.shape[2]])

    fitness = fitness.detach().cpu().numpy().ravel()
    for i in range(child_size):
        idx = np.random.choice(np.arange(pop.shape[0]), size=2, replace=True,
                               p=(fitness) / (fitness.sum()))
        child = pop[idx][0]
        mother = pop[idx][1]
        cross_points = np.random.randint(low=1, high=pop.shape[2])  # Randomly generated intersection points
        child[:, cross_points:] = mother[:, cross_points:]  # The child obtains the mother's genes located behind the intersection point
        new_pop[i,...] = child

    return new_pop
def reinitial_same_pop(pop, candicate=None):
    name_dict = {}

    for i,xx in enumerate(pop):
        xx_dict = {}
        name = seq_to_name(xx)
        xx_dict['%s'%name]=xx
        if name_dict.keys().isdisjoint(xx_dict.keys()):
            name_dict['%s'%name] = xx
        else:
            pop[i,...]=generate_initial_group(1,pop.shape[2])
    return pop
def seq_to_name(seq):
    ind = torch.where(seq == torch.tensor(1))
    seq = 'A' * seq.shape[1]
    seq = list(seq)
    for base_n, i in zip(ind[0], ind[1]):
        if base_n == 0:
            seq[i] = 'A'
        elif base_n == 1:
            seq[i] = 'C'
        elif base_n == 2:
            seq[i] = 'G'
        elif base_n == 3:
            seq[i] = 'T'
    return seq
def detect_same_pop(pop, candicate=None):

    if candicate!=None:
        if candicate.tolist() in pop.tolist():
            return False
        else:
            return True

    else:
        name_dict = {}
        for i, xx in enumerate(pop):
            xx_dict = {}
            name = seq_to_name(xx)
            xx_dict['%s' % name] = i
            if name_dict.keys().isdisjoint(xx_dict.keys()):
                name_dict['%s' % name] = i
            else:
                print('%d-%d repeat'% (name_dict['%s' % name], i))

def select(pop, fitness, pop_size, ind=None, top_fitness_pop_size=None, mutation_probability=None):  # nature selection wrt pop's fitness
    fitness = fitness.detach().cpu().numpy().ravel()
    idx = np.random.choice(np.arange(pop.shape[0]), size=pop_size, replace=False,
                           p=(fitness) / (fitness.sum()))
    # num1 = np.where((ind[idx] >=top_fitness_pop_size)&(ind[idx]<mutation_probability.shape[0]))[0].shape[0]
    # num2 = np.where(ind[idx] < top_fitness_pop_size)[0].shape[0]
    # num3 = np.where(ind[idx] >= mutation_probability.shape[0])[0].shape[0]
    # print(num2,num1,num3 )
    return pop[idx]