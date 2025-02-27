import os
from re import S
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import librosa
import yaml
from torch.utils.mobile_optimizer import optimize_for_mobile
import typing

from model import RawNet

config_path = "model_config_RawNet.yaml"
model_path = "models/model_LA_CCE_100_64_0.0001/epoch_71.pth"
device = 'cpu' 
with open(config_path, 'r') as f_yaml:
    parser1 = yaml.safe_load(f_yaml)

model = RawNet(parser1['model'], device).to(device)
model.load_state_dict(torch.load(model_path,map_location=device))

class BTSDetect(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    
    def pad(self, x, max_len: int = 16000):
        x_len = x.shape[0]
        # print(x_len)
        if (x_len>=max_len):
            pad_x =  x[:max_len]
        else:
            num_repeats = int(max_len/x_len)+1
            pad_x = x.repeat(1,num_repeats)[0][:max_len]
        # print(pad_x)
        return pad_x.unsqueeze(0)
    
    def forward(self, wavforms: Tensor):
        wav_padded = self.pad(wavforms)
        print(wav_padded.shape)
        logits, _ = self.model(wav_padded)
        # convert to probability
        logits = torch.softmax(logits, dim=1)
        print(logits)
        return logits[0][0]

_model = BTSDetect(model)
# Sanity check
file_path = "/dataa/phucdt/bio/test_data/LA_E_5085671.flac"
data, y = librosa.load(file_path, sr=16000)
res = _model(Tensor(data))
print(res)


# # Apply quantization / script / optimize for motbile
_model.eval()
print("DEBUG1")
quantized_model = torch.quantization.quantize_dynamic(
    _model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
print("DEBUG2")
scripted_model = torch.jit.script(quantized_model)
print("DEBUG3")
optimized_model = optimize_for_mobile(scripted_model)
print("DEBUG4")

# Sanity check
file_path = "/dataa/phucdt/bio/test_data/LA_E_5085671.flac"
data, y = librosa.load(file_path, sr=16000, mono=True)
print(Tensor(data))
res = optimized_model(Tensor(data))
print("DEBUG5")
print(res)
# print('Result:', optimized_model(Tensor(data)))
optimized_model._save_for_lite_interpreter("btsdetect_cnn.ptl")