import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def make_target(labels, n=2, long_tensor=False, device=device):
    if(long_tensor):
        return torch.LongTensor(labels).to(device)
    #Target is 1-hot vector
    target = torch.zeros(labels.shape[0],n).to(device)
    for label_i, label in enumerate(labels):
        target[label_i][int(label)] = 1
    return target

def features_to_num(list_features, val_to_match):
    append_list = []
    for fval in list_features:
        if fval in val_to_match:
            append_list.append(1)
        else:
            append_list.append(0)
    return append_list

def smooth_distances(distances_matrix):
    distances_matrix = (distances_matrix*100)+.000001
    for i in range(0, distances_matrix.shape[0]): #distances between the same data point should be 0
        distances_matrix[i][i] = 0.0 
    return distances_matrix

def shuffle_turk_distances(matrix_to_shuffle, shuffle_val):
    #np.random.seed(0)
    if(shuffle_val == 0.0):
        return matrix_to_shuffle
    dists = matrix_to_shuffle.copy()
    shuffled_dists= matrix_to_shuffle.copy()
    np.random.shuffle(shuffled_dists)
    prop_shuffle = int(dists.shape[0]*shuffle_val)
    for i in range(0, prop_shuffle):
        dists[i] = shuffled_dists[i]
    return dists 

def initialize(model, init_model, warm_start=True):
    if(warm_start):
        w1 = init_model.linear.weight.clone().detach()
        b1 = init_model.linear.bias.clone().detach()
        model.linear.weight = nn.Parameter(w1.clone()) 
        model.linear.bias = nn.Parameter(b1.clone())
    return model 