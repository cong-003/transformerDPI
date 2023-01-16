import torch
import numpy as np
import random
import os
import time
from model import *
import timeit
import pickle

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    DATASET = "GPCR_test"
    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    dir_input = ('../dataset/' + DATASET + '/word2vec_30/')
    compounds = load_tensor(dir_input + 'compounds', torch.FloatTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.FloatTensor)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)

    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(compounds, adjacencies, proteins, interactions))
    dataset = shuffle_dataset(dataset, 1234)

    """ create model ,trainer and tester """
    protein_dim = 100
    atom_dim = 34
    hid_dim = 64
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    batch = 64
    lr = 1e-4
    weight_decay = 1e-4
    decay_interval = 5
    lr_decay = 1.0
    iteration = 300
    kernel_size = 7

    encoder1 = Encoder(protein_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device)
    encoder2 = Encoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device)
    model = Predictor(encoder1, encoder2, decoder, device)   
    
    # load trained model
    model.load_state_dict(torch.load("output/model/lr=1e-4,dropout=0.1,weight_decay=1e-4,kernel=7,n_layer=3,batch=64"))
    model.to(device)
    print(model)
    tester = Tester(model)

    """Output files."""
    file_AUCs = 'output/result/model_eval_AUCs--lr=1e-4,dropout=0.1,weight_decay=1e-4,kernel=7,n_layer=3,batch=64'+ '.txt'
    AUC = ('AUC_dev\tPRC_dev\tLoss_test')
    with open(file_AUCs, 'w') as f:
        f.write(AUC + '\n')

    """Start evaluating."""
    print('Evaluating...')
    print(AUC)
 
    AUCs = tester.test(dataset)
    tester.save_AUCs(AUCs, file_AUCs)

    print('\t'.join(map(str, AUCs)))



