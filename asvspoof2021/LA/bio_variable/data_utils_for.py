import os
from re import S
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset

#PHUCDT
from biosegment import wav2bio

___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"


def genSpoof_list(dir_meta, is_train=False, is_eval=False, tts_only=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
            key,subset,label = line.strip().split(' ')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list

    elif(is_eval):
        for line in l_meta:
            key,subset,label = line.strip().split(' ')
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            key,subset,label = line.strip().split(' ')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class Dataset_for(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)'''

        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        key = self.list_IDs[index]
        # X, fs = librosa.load(self.base_dir+'flac/'+key+'.flac', sr=16000)
        X, fs = librosa.load(self.base_dir + "/" + key, sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        
        bio_inp, bio_length = self.get_Bio(key, X_pad, fs)
        
        y = self.labels[key]
        
        return x_inp, bio_inp, bio_length, y
 
    def get_Bio(self, filename, X_pad, fs):
        
        feat_name = "./feats/add/" + filename

        if os.path.exists(feat_name):
            bio_inp = torch.load(feat_name)
            bio_length = bio_inp.size(0)
            # print(bio_length)
            
        else:
            bio = wav2bio(X_pad, fs)
            bio_length = len(bio)
            bio_inp = torch.IntTensor(bio)
            torch.save(bio_inp, feat_name)
        return bio_inp, bio_length
    

class Dataset_for_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
           '''

        self.list_IDs = list_IDs
        self.base_dir = base_dir

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        key = self.list_IDs[index]
        # X, fs = librosa.load(self.base_dir+'flac/'+key+'.flac', sr=16000)
        X, fs = librosa.load(self.base_dir + "/" + key, sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        
        bio_inp, bio_length = self.get_Bio(key, X_pad, fs)
        
        return x_inp, bio_inp, bio_length, key
    def get_Bio(self, filename, X_pad, fs):
        
        feat_name = "./feats/add/" + filename

        if os.path.exists(feat_name):
            bio_inp = torch.load(feat_name)
            bio_length = bio_inp.size()[0]
            
        else:
            bio = wav2bio(X_pad, fs)
            bio_length = len(bio)
            bio_inp = torch.IntTensor(bio)
            torch.save(bio_inp, feat_name)
        return bio_inp, bio_length
