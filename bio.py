
from breathing.biosegment import wav2bio
from breathing import RawNet
import yaml
import os
from re import S
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
here = os.path.dirname(os.path.abspath(__file__))

class bio():
    def __init__(self, device):
        self.config_path = here + "/breathing/configs/model_config_RawNet_Trans_32concat.yaml"
        self.model_path = here + "/breathing/models/trans_32concat/epoch_91.pth"
        self.cache_dir = here + "/breathing/cache/"
        self.device = device

        with open(self.config_path, 'r') as f_yaml:
            parser1 = yaml.safe_load(f_yaml)
        self.model = RawNet(parser1['model'], self.device).to(device)
        self.model.load_state_dict(torch.load(self.model_path,map_location=self.device))
        self.model.eval()
    
    def get_Bio(self, X_pad, fs, file_path):
        
        bio = wav2bio(X_pad, fs)
        # bio_length = len(bio)
        bio_inp = torch.IntTensor(bio)
        bio_length = torch.IntTensor([len(bio)])
        bio = wav2bio(X_pad, fs)
        feat_name = self.cache_dir + os.path.basename(file_path)
        torch.save(bio_inp, feat_name)
        return bio_inp, bio_length

    def pad(self, x, max_len=64600):
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        # need to pad
        num_repeats = int(max_len / x_len)+1
        padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
        return padded_x

    def parse_input(self, file_path):
        cut = 64600  # take ~4 sec audio (64600 samples)
        X, fs = librosa.load(file_path, sr=16000)
        X_pad = self.pad(X, cut)
        x_inp = Tensor(X_pad)
        feat_name = self.cache_dir + os.path.basename(file_path)
        if (os.path.exists(feat_name)):
            bio_inp = torch.load(feat_name)
            bio_length = torch.IntTensor([bio_inp.size(0)])
        else:
            bio_inp, bio_length = self.get_Bio(X_pad, fs, file_path)

        return x_inp.unsqueeze(0).to(self.device), bio_inp.unsqueeze(0).to(self.device), bio_length.to(self.device)
    
    def get_output(self, file_path):
        x_inp, bio_inp, bio_length = self.parse_input(file_path)
        # print(bio_length)
        # print(x_inp)
        # print(bio_inp)
        return self.model(x_inp, bio_inp, bio_length)
    

if __name__ == '__main__':
    file_path = "/root/dataset/ASVspoof/LA/ASVspoof2019_LA_train/flac/LA_T_4427701.flac"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    teacher = bio(device)
    
    a, b = teacher.get_output(file_path)
    print(a,b)
    
    