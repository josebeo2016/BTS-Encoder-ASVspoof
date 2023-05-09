import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from AdvModel import bio


___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

""" backup
def genSpoof_list(dir_meta, is_train=False, is_eval=False, tts_only=True):
    
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
            _, key, _, att, label = line.strip().split(' ')
            if (tts_only):
                if((att=="A05") or (att=="A06")):
                    continue
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list

    elif(is_eval):
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, att, label = line.strip().split(' ')
            if (tts_only):
                if((att=="A05") or (att=="A06")):
                    continue
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list
"""
def genSpoof_list(dir_meta, is_train=False, is_eval=False, tts_only=False):
    
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
            _, key, _, att, label = line.strip().split(' ')
            if (tts_only):
                if((att=="A05") or (att=="A06")):
                    continue
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list

    elif(is_eval):
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, att, label = line.strip().split(' ')
            if (tts_only):
                if((att=="A05") or (att=="A06")):
                    continue
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
			

class Dataset_ASVspoof2019_train(Dataset):
	def __init__(self, list_IDs, labels, base_dir):
            '''self.list_IDs	: list of strings (each string: utt key),
               self.labels      : dictionary (key: utt key, value: label integer)'''
               
            self.list_IDs = list_IDs
            self.labels = labels
            self.base_dir = base_dir
            self.teacher_res_dir = "/root/biological/AdvAttacksASVspoof/model/breathing_result/"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.teacher = bio(device)

	def __len__(self):
           return len(self.list_IDs)


	def __getitem__(self, index):
            self.cut=64600 # take ~4 sec audio (64600 samples)
            key = self.list_IDs[index]
            X,fs = librosa.load(self.base_dir+'flac/'+key+'.flac', sr=16000) 
            X_pad= pad(X,self.cut)
            x_inp= Tensor(X_pad)
            y = self.labels[key]
            
            # load teacher score
            out_file = self.teacher_res_dir + key + ".flac"
            logit, emb = self.teacher.get_output(out_file)
            
            return x_inp, y, logit, emb
            
            
class Dataset_ASVspoof2021_eval(Dataset):
	def __init__(self, list_IDs, base_dir):
            '''self.list_IDs	: list of strings (each string: utt key),
               '''
               
            self.list_IDs = list_IDs
            self.base_dir = base_dir
            

	def __len__(self):
            return len(self.list_IDs)


	def __getitem__(self, index):
            self.cut=64600 # take ~4 sec audio (64600 samples)
            key = self.list_IDs[index]
            X, fs = librosa.load(self.base_dir+'flac/'+key+'.flac', sr=16000)
            X_pad = pad(X,self.cut)
            x_inp = Tensor(X_pad)
            return x_inp,key           
           
            
class Dataset_ASVspoof2019_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
           '''

        self.list_IDs = list_IDs
        self.base_dir = base_dir

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        keys = self.list_IDs[index].strip().split(' ')
        # key = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+'flac/'+keys[1]+'.flac', sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        
        return x_inp, self.list_IDs[index]            

                
                
                



