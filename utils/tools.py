import numpy as np
import pandas as pd
from Bio import SeqIO
import yaml
import os

dic = {"A": "T", "T": "A", "C": "G", "G": "C"}
# Convert one hot encoding into DNA sequence
def one_hot_to_dna(one_hot_matrix):
    mapping = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    dna_sequences = []

    if(len(one_hot_matrix.shape)==2):
        return ''.join([mapping[np.argmax(base)] for base in one_hot_matrix.T])

    for sequence in one_hot_matrix:
        dna_sequence = ''.join([mapping[np.argmax(base)] for base in sequence.T])
        dna_sequences.append(dna_sequence)

    return dna_sequences

def rev_complement(seq):
    seq2 = ""
    for i in seq:
        seq2 = dic[i] + seq2

    return seq2
def convert_fasta_to_pd(fasta_file):
    records = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        chromosome = record.id
        sequence = str(record.seq)
        start = 1
        end = len(sequence)
        records.append([chromosome.split(':')[0], chromosome.split(':')[1].split('-')[0], chromosome.split(':')[1].split('-')[1], sequence,rev_complement(sequence)])

    df = pd.DataFrame(records, columns=['chr_name', 'start', 'end', 'sequence','rev_sequence'])
    return df
def run_cmd(cmd_str='', echo_print=0):

    from subprocess import run
    if echo_print == 1:
        print('\nExecute cmd instruction="{}"'.format(cmd_str))
    run(cmd_str, shell=True)
def save_dict_to_yaml(dict_value: dict, save_path: str):
    """save dict to yaml"""
    with open(save_path, 'w') as file:
        file.write(yaml.dump(dict_value, allow_unicode=True))
 
 
def read_yaml_to_dict(yaml_path: str):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
    return dict_value


def generate_intervals(x:list):
    intervals=[]
    for i in range(len(x) - 1):
        intervals.append((x[i],x[i+1]))
    return intervals


def count_numbers_in_intervals(intervals, data_array):

    interval_counts = {interval: 0 for interval in intervals}
    

    for number in data_array:
        for interval in intervals:
            if interval[0] <= number <= interval[1]:
                interval_counts[interval] += 1
                
    return interval_counts

def binary_search_interval(intervals, number):

    left, right = 0, len(intervals) - 1
    
    while left <= right:
        mid = (left + right) // 2
        start, end = intervals[mid]
        
        if start <= number <= end:
            return mid
        elif number < start:
            right = mid - 1
        else:
            left = mid + 1
    
    return -1

def create_dir(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except FileExistsError:
            print(f"{dir} exists")