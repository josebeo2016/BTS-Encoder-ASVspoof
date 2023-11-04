import numpy as np
import pandas as pd
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch import Tensor
from tqdm import tqdm
import sys
from auditok import DataSource, DataValidator
import os
import yaml
import argparse
logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
"""
Labels1d
"""
BIOTYPE = {
    "silence":0,
    "breath":1,
    "speech":2
}
# https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=3, stride=1, n_channel=32, h_in=30, w_in=120):
        super().__init__()
        self.conv1 = nn.Conv2d(n_input, n_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channel)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(n_channel, n_channel, kernel_size=(2,2))
        self.bn2 = nn.BatchNorm2d(n_channel)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(n_channel, 2 * n_channel, kernel_size=(1,1), padding=(1,1))
        self.bn3 = nn.BatchNorm2d(2 * n_channel)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(2 * n_channel, 2 * n_channel, kernel_size=(1,1), padding=(1,1))
        self.bn4 = nn.BatchNorm2d(2 * n_channel)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=(2, 2))
        
        self.fc1 = nn.Linear( 2 * int(w_in/h_in) * n_channel, 2* n_channel)
        self.fc2 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        # input shape: (batch_size, n_time, n_channel)
        x = x.unsqueeze(1) # (batch_size, 1, n_time, n_channel)
        x = x.permute(0, 1, 3, 2) # (batch_size, 1, n_channel, n_time)
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = self.avgpool(x)
        # flatten
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        return out
    
    def num_flat_features(self, x):
        size = x.size()[1:] # remove batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    def predict(self, x):
        logits = self.forward(x)
        return logits.argmax(dim=-1)

class LCNN(nn.Module):
    def __init__(self, num_conv_layers=4, num_channels=32, kernel_size=3, stride = 1, num_linear_layers=2, hidden_size=128, num_classes=3, h_in=30, w_in =120):
        super(LCNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        padding = (kernel_size-stride)//2
        dilation = 1
        # first conv layer
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size, stride=stride, padding=padding)
        for _ in range(num_conv_layers):
            self.conv_layers.append(nn.Conv2d(num_channels, num_channels, kernel_size, stride=stride, padding=padding))
        
        h_out = int((h_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
        w_out = int((w_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
        # Linear layers
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(num_channels * h_out * w_out, hidden_size))
        for _ in range(num_linear_layers - 1):
            self.fc_layers.append(nn.Linear(hidden_size, hidden_size))
        self.fc_layers.append(nn.Linear(hidden_size, num_classes))

        # Activation functions
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        for conv_layer in self.conv_layers:
            x = self.relu(conv_layer(x))
        x = torch.flatten(x, 1)
        for fc_layer in self.fc_layers[:-1]:
            x = self.relu(fc_layer(x))
        x = self.fc_layers[-1](x)
        x = F.log_softmax(x, dim=1)
        return x



class CNNClassifier():
    def __init__(self, model_path, config, device='cpu'):
        
        self.model = globals()[config['model']['name']](**config['model']['params'])
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        self.feat_len = config['model']['feat_len']
        self.padding_type = config['model']['padding_type']
    
    def predict(self, x, class_weight: list=[1,1,1]):
        # convert list to tensor
        class_weight = torch.tensor(class_weight).to(self.device)
        with torch.no_grad():
            x = pad(x,padding_type=self.padding_type,max_len=self.feat_len)
            x = x.to(self.device)
            # add batch dim
            x = x.unsqueeze(0)
            logits = self.model(x)
            weighted_logits = logits * class_weight.unsqueeze(0)
            out = weighted_logits.argmax(dim=-1).cpu().numpy()[0]
        torch.cuda.empty_cache()
        return out
    def predict_batch(self, data_list: list, class_weight: list=[1,1,1], batch_size = 32):
        # convert list to tensor
        class_weight = torch.tensor(class_weight).to(self.device)
        with torch.no_grad():
            data = FeatLoaderEval(data_list, feat_len=self.feat_len)
            data_loader = DataLoader(data, batch_size=32, shuffle=False)
            res = []
            for batch in data_loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                weighted_logits = logits * class_weight.unsqueeze(0)
                out = weighted_logits.argmax(dim=-1)
                res.append(out)

        torch.cuda.empty_cache()
        return torch.cat(res, dim=0).cpu().tolist()
    
class ClassifierValidator(DataValidator):
    
    def __init__(self, classifier, target):
        """
        classifier: a GMMClassifier object
        target: string
        """
        self.classifier = classifier
        self.target = BIOTYPE[target]
        
    def is_valid(self, data):
        
        r = self.classifier.predict(data)
        # if (r[0][0] == self.target):
        # print(r[0][0],r[0][1], self.target)
        return r == self.target
    
class VectorDataSource(DataSource):
     
    def __init__(self, data, scope=0):
        self.scope = scope
        self._data = data
        self._current = 0
    
    def read(self):
        if self._current >= len(self._data):
            return None
        
        start = self._current - self.scope
        if start < 0:
            start = 0
            
        end = self._current + self.scope + 1
        
        self._current += 1
        return self._data[start : end]
    
    def set_scope(self, scope):
        self.scope = scope
            
    def rewind(self):
        self._current = 0

def label_to_index(label: str):
    return BIOTYPE[label]

def repeat_padding_Tensor(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[0]))
    spec = spec.repeat(mul, 1)[:ref_len, :]
    return spec

def zero_padding_Tensor(spec, ref_len):
    if spec.shape[0] < ref_len:
        spec = torch.cat([spec, torch.zeros(ref_len - spec.shape[0], spec.shape[1])])
        
    else:
        spec = spec[:ref_len, :]
    return spec

def pad(feat, padding_type = 'zero',max_len=30):
    """
    padding_type: zero or repeat
    """
    assert padding_type in ['zero', 'repeat']
    this_feat_len = feat.shape[0]
    featureTensor = Tensor(feat)
    # padding
    if this_feat_len > max_len:
        startp = np.random.randint(this_feat_len - max_len)
        featureTensor = featureTensor[startp:startp + max_len, :]

    if this_feat_len < max_len:
        if padding_type == 'repeat':
            featureTensor = repeat_padding_Tensor(featureTensor, max_len)
        if padding_type == 'zero':
            featureTensor = zero_padding_Tensor(featureTensor, max_len)
    return featureTensor

class FeatLoader(Dataset):
    def __init__(self, data: pd.DataFrame, padding_type: str="zero", feat_len = 30):
        self.data = data
        self.feat_len = feat_len
        self.padding_type = padding_type
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        feat = self.data['features'][index]
        feat = pad(feat,padding_type=self.padding_type,max_len=self.feat_len)
        y = label_to_index(self.data['class'][index])
        return feat, y
    


class FeatLoaderEval(Dataset):
    def __init__(self, data: list, padding_type: str="zero", feat_len = 30):
        self.data = data
        self.feat_len = feat_len
        self.padding_type = padding_type
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        feat = self.data[index]
        feat = pad(feat,padding_type=self.padding_type,max_len=self.feat_len)
        return feat


def train(model, epoch, log_interval, device):
    model.train()
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    criterion = nn.CrossEntropyLoss()
    for batch_feats, batch_y in train_loader:
        
        batch_size = batch_feats.size(0)
        num_total += batch_size
        batch_feats = batch_feats.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        batch_out = model(batch_feats)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        
        # Calculate loss
        batch_loss = criterion(batch_out, batch_y)
        running_loss += (batch_loss.item() * batch_size)
        
        # print training stats
        if ii % 10 == 0:
            sys.stdout.write('\r \t {:.2f}'.format(
                (num_correct/num_total)*100))
            
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(batch_loss.item())
def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def test(model, epoch, device):
    model.eval()
    num_correct = 0
    num_total = 0
    for data, target in test_loader:
        data = data.to(device)
        data_size = data.size(0)
        num_total += data_size
        
        target = target.view(-1).type(torch.int64).to(device)

        output = model(data)

        _, batch_pred = output.max(dim=1)
        num_correct += (batch_pred == target).sum(dim=0).item()

        # update progress bar
        pbar.update(pbar_update)

    print(f"\nTest Epoch: {epoch}\tAccuracy: {num_correct}/{num_total} ({100. * num_correct / num_total:.0f}%)\n")
    return num_correct / num_total

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        prog="CNN_breath.py", description="Training CNN for breath detection."
    )

    parser.add_argument(
        "feat_file",
        type=str,
        help="a path to a hdf5 file containing features and labels",
    )

    parser.add_argument(
        "out_dir",
        type=str,
        help="a path to a directory where the output models will be saved",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="cnn_config.yaml",
        help="a path to a yaml config file for feature extraction",
    )

    args = parser.parse_args()

    feats = pd.read_hdf(args.feat_file)
    train_set = feats.sample(frac=0.8, random_state=1234, ignore_index=True)
    test_set = feats.drop(train_set.index)
    test_set.reset_index(drop=True, inplace=True)
    
    # load config
    config = yaml.safe_load(open(args.config, "r"))    
    feat_len = config["model"]["feat_len"]

    # load data

    train_loader = DataLoader(FeatLoader(train_set), batch_size=config['model']['batch_size'], shuffle=True, drop_last = True)
    test_loader = DataLoader(FeatLoader(test_set), batch_size=config['model']['batch_size'], shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print(test_loader.dataset[0][0])

    # load model
    model = globals()[config['model']['name']](**config['model']['params'])
    model.to(device)
    model.train()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['model']['lr'], weight_decay=config['model']['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    log_interval = 20
    n_epoch = config['model']['n_epoch']

    pbar_update = 1 / (len(train_loader) + len(test_loader))
    losses = []

    with tqdm(total=n_epoch) as pbar:
        for epoch in range(1, n_epoch + 1):
            train(model, epoch, log_interval, device)
            test(model, epoch, device)
            scheduler.step()
    print('Finished Training! Save to {}'.format(os.path.join(args.out_dir, 'cnn.pth')))
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'cnn.pth'))

        