import argparse
from statistics import mode
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
from data_utils_fuse import genSpoof_list, genSpoof_list_custom, Dataset_for, Dataset_for_eval
from model import Model
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from tqdm import tqdm

__author__ = "PHUCDT"
__credits__ = ["Jose Patino", "Massimiliano Todisco", "Jee-weon Jung", "Hemlata Tak"]

class EarlyStop:
    def __init__(self, patience=5, delta=0, init_best=60, save_dir=''):
        self.patience = patience
        self.delta = delta
        self.best_score = init_best
        self.counter = 0
        self.early_stop = False
        self.save_dir = save_dir

    def __call__(self, score, model, epoch):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print("Best epoch: {}".format(epoch))
            self.best_score = score
            self.counter = 0
            # save model here
            torch.save(model.state_dict(), os.path.join(
                self.save_dir, 'epoch_{}.pth'.format(epoch)))

def evaluate_accuracy(dev_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_bio, bio_lengths, batch_y in tqdm(dev_loader):
            
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_bio = batch_bio.to(device)
            bio_lengths = bio_lengths.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out, _ = model(batch_x, batch_bio, bio_lengths)
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()
    return 100 * (num_correct / num_total)

def train_epoch(train_loader, model, lr,optim, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    model.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.5, 0.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    #PHUCDT 
    for batch_x, batch_bio, bio_lengths, batch_y in tqdm(train_loader):
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        
        batch_x = batch_x.to(device)
        batch_bio = batch_bio.to(device)
        bio_lengths = bio_lengths.to(device)
        
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out, _ = model(batch_x,batch_bio, bio_lengths)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        if ii % 10 == 0:
            sys.stdout.write('\r \t {:.2f}'.format(
                (num_correct/num_total)*100))
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
       
    running_loss /= num_total
    train_accuracy = (num_correct/num_total)*100
    return running_loss, train_accuracy

def produce_emb_file(dataset, model, device, save_path, batch_size=10):
    data_loader = DataLoader(dataset, batch_size, shuffle=False, drop_last=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    model.is_train = True

    fname_list = []
    key_list = []
    score_list = []
    with torch.no_grad():
        for batch_x, batch_bio, bio_lengths, utt_id in tqdm(data_loader):
            fname_list = []
            score_list = []  
            pred_list = []
            batch_size = batch_x.size(0)
            
            batch_x = batch_x.to(device)
            bio_lengths = bio_lengths.to(device)
            batch_bio = batch_bio.to(device)
            batch_out, batch_emb = model(batch_x, batch_bio, bio_lengths)
            
            score_list.extend(batch_out.data.cpu().numpy().tolist())
            # add outputs
            fname_list.extend(utt_id)

            # save_path now must be a directory
            # make dir if not exist
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # Then each emb should be save in a file with name is utt_id
            for f, emb in zip(fname_list,batch_emb):
                # normalize filename
                f = f.split('/')[-1].split('.')[0] # utt id only
                save_path_utt = os.path.join(save_path, f)
                np.save(save_path_utt, emb.data.cpu().numpy())
            
            # score file save into a single file
            with open(os.path.join(save_path, "scores.txt"), 'a+') as fh:
                for f, cm in zip(fname_list,score_list):
                    fh.write('{} {} {}\n'.format(f, cm[0], cm[1]))
            fh.close()   
    print('Scores saved to {}'.format(save_path))

def produce_bio_extract_file_2019(
    dataset,
    model,
    device,
    save_path):
    """Perform evaluation and save the score to a file"""
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False)
    model.eval()
    with torch.no_grad():
        for batch_x, batch_bio, bio_lengths, keys in tqdm(data_loader):
            # fname_list = []
            # score_list = []  
            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)
            bio_lengths = bio_lengths.to(device)
            batch_bio = batch_bio.to(device)
            _, batch_bio_out = model(batch_x, batch_bio, bio_lengths)

                        
            # add outputs
            # fname_list.extend(keys)
            # score_list.extend(batch_score.tolist())
            

            for fn, sco in zip(keys, batch_bio_out.tolist()):
                _, utt_id, _, src, key = fn.strip().split(' ')
                torch.save(sco, save_path + "/" + utt_id)
                # assert fn == utt_id
            

    # assert len(trial_lines) == len(fname_list) == len(score_list)
    # with open(save_path, "w") as fh:
    #     for fn, sco, trl in zip(fname_list, score_list, trial_lines):
    #         _, utt_id, _, src, key = trl.strip().split(' ')
    #         assert fn == utt_id
    #         fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("bio saved to {}".format(save_path))

def produce_evaluation_file_2019(
    dataset,
    model,
    device,
    save_path):
    """Perform evaluation and save the score to a file"""
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)
    model.eval()
    
    for batch_x, batch_bio, bio_lengths, keys in data_loader:
        # fname_list = []
        # score_list = []  
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        bio_lengths = bio_lengths.to(device)
        batch_bio = batch_bio.to(device)
        batch_out, _ = model(batch_x, batch_bio, bio_lengths)
        batch_score = (batch_out[:, 1]
                       ).data.cpu().numpy().ravel()
        # add outputs
        # fname_list.extend(keys)
        # score_list.extend(batch_score.tolist())
        
        with open(save_path, "a+") as fh:
            for fn, sco in zip(keys, batch_score.tolist()):
                _, utt_id, _, src, key = fn.strip().split(' ')
                # assert fn == utt_id
                fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
        fh.close()
    # assert len(trial_lines) == len(fname_list) == len(score_list)
    # with open(save_path, "w") as fh:
    #     for fn, sco, trl in zip(fname_list, score_list, trial_lines):
    #         _, utt_id, _, src, key = trl.strip().split(' ')
    #         assert fn == utt_id
    #         fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))

def produce_evaluation_file(dataset, model, device, save_path, batch_size):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    with torch.no_grad():
        for batch_x, batch_bio, bio_lengths, utt_id in tqdm(data_loader):
            fname_list = []
            score_list = []  
            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)
            bio_lengths = bio_lengths.to(device)
            batch_bio = batch_bio.to(device)
            batch_out, _ = model(batch_x, batch_bio, bio_lengths)
            # batch_score = (batch_out[:, 1]
            #             ).data.cpu().numpy().ravel()
            # add outputs
            fname_list.extend(utt_id)
            score_list = batch_out.data.cpu().numpy().tolist()
            
            with open(save_path, 'a+') as fh:
                for f, cm in zip(fname_list,score_list):
                    fh.write('{} {} {}\n'.format(f, cm[0], cm[1]))
            fh.close()   
    print('Scores saved to {}'.format(save_path))
    
def produce_prediction_file(dataset, model, device, save_path, batch_size):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    with torch.no_grad():
        for batch_x, batch_bio, bio_lengths, utt_id in tqdm(data_loader):
            fname_list = []
            score_list = []  
            prob_list = []
            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)
            bio_lengths = bio_lengths.to(device)
            batch_bio = batch_bio.to(device)
            batch_out, _ = model(batch_x, batch_bio, bio_lengths)
            _, batch_pred = batch_out.max(dim=1)
            batch_score = (batch_out[:, 1]
                        ).data.cpu().numpy().ravel()
            # add outputs
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())
            # calculate probability
            batch_prob = nn.Softmax(dim=1)(batch_out)
            prob_list.extend(batch_prob.tolist())
            with open(save_path, 'a+') as fh:
                for f, cm, prob in zip(fname_list,score_list, prob_list):
                    fh.write('{} {} {}\n'.format(f, cm, prob[1]))
            fh.close()   
    print('Scores saved to {}'.format(save_path))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset

    parser.add_argument('--database_path', type=str, default='/dataa/Dataset/ASVspoof/LA/', help='path to database')
    '''
    % database_path/
    %   |- protocol.txt
    %   |- audio path
    
    Protocol has 3 columns: filename, subset, label
    LA_T_1000001.wav train bonafide
    LA_D_1000002.wav dev spoof
    '''


    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--track', type=str, default='LA',choices=['LA', 'PA','DF'], help='LA/PA/DF')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config yaml file')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode for all data with protocol.txt')
    parser.add_argument('--bio_extract', action='store_true', default=False,
                        help='extract bio feature')
    parser.add_argument('--emb', action='store_true', default=False,
                        help='extract embedding and score for fuse model') 
    parser.add_argument('--custom', action='store_true', default=False,
                        help='custom eval set, care about the first column of protocol.txt')
    parser.add_argument('--predict', action='store_true', default=False,
                        help='get the predicted label instead of score')
    parser.add_argument('--is_eval', action='store_true', default=False,help='eval database')
    parser.add_argument('--eval_2019', action='store_true', default=False, help='eval on ASVspoof2019 eval set, split follow attack type')
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)') 
    
    parser.add_argument('--print_param', action='store_true', default=False, help='print the weight params of the trained model')
    

    

    if not os.path.exists('models'):
        os.mkdir('models')
    
    if not os.path.exists('feats'):
        os.mkdir('feats')
    args = parser.parse_args()
    
    # dir_yaml = os.path.splitext('model_config_RawNet')[0] + '.yaml'
    dir_yaml = args.config

    with open(dir_yaml, 'r') as f_yaml:
            parser1 = yaml.safe_load(f_yaml)
            
    #make experiment reproducible
    set_random_seed(args.seed, args)
    
    #define model saving path
    model_tag = 'model_{}_{}_{}_{}'.format(
        args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    #model 
    model = Model(parser1['model'], device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model =(model).to(device)
    
    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    # support multiple GPUs
    
    if args.model_path:
        try:
            model.load_state_dict(torch.load(args.model_path,map_location=device))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
        except:
            print('DataParallel enabled')
            model = Model(parser1['model'], device)
            model =(model).to(device)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model.load_state_dict(torch.load(args.model_path,map_location=device))
            
        print('Model loaded : {}'.format(args.model_path))
    
    else:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print('Model initialized')
        
    # print param
    if args.print_param:
        print(model)
        print(model.fc2_gru.weight)
        torch.save(model.fc2_gru.weight, '/root/biological/last_weight'+args.config.replace(".yaml",".pt"))
        
        # for param in model.parameters():
        #     print(param)
        sys.exit(0)
        
    
    #evaluation 
    if args.eval:
        if args.custom:
            file_eval = genSpoof_list_custom( dir_meta =  os.path.join(args.database_path,'protocol.txt'))
        else:
            file_eval = genSpoof_list( dir_meta =  os.path.join(args.database_path,'protocol.txt'),is_train=False, is_eval=True, is_dev=False)
        print('no. of eval trials',len(file_eval))
        eval_set=Dataset_for_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path))
        if (args.emb):
            produce_emb_file(eval_set, model, device, args.eval_output,batch_size=args.batch_size)
        elif (args.predict):
            produce_prediction_file(eval_set, model, device, args.eval_output,batch_size=args.batch_size)
        else:
            produce_evaluation_file(eval_set, model, device, args.eval_output,batch_size=args.batch_size)
        sys.exit(0)

    #PHUCDT

    # define train dataloader
    d_label_trn,file_train = genSpoof_list( dir_meta = os.path.join(args.database_path,'protocol.txt'),is_train=True,is_eval=False)
    print('no. of training trials',len(file_train))
    
    train_set=Dataset_for(list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,drop_last = True)
    
    del train_set,d_label_trn
    

    # define validation dataloader

    d_label_dev,file_dev = genSpoof_list( dir_meta =  os.path.join(args.database_path,'protocol.txt'),is_train=False,is_eval=False, is_dev=True)
    print('no. of validation trials',len(file_dev))

    dev_set = Dataset_for(list_IDs = file_dev,
		labels = d_label_dev,
		base_dir = os.path.join(args.database_path))
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False)
    del dev_set,d_label_dev

    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    best_acc = 99
    early_stopping = EarlyStop(patience=10, delta=0.01, init_best=best_acc, save_dir=model_save_path)
    for epoch in range(num_epochs):
        running_loss, train_accuracy = train_epoch(train_loader,model, args.lr,optimizer, device)
        valid_accuracy = evaluate_accuracy(dev_loader, model, device)
        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        writer.add_scalar('valid_accuracy', valid_accuracy, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - {} - {:.2f} - {:.2f}'.format(epoch,
                                                   running_loss, train_accuracy, valid_accuracy))
        
        # early stopping
        early_stopping(valid_accuracy, model, epoch)
        if early_stopping.early_stop:
            print("Early stopping activated.")
            
            break
        # if valid_accuracy > best_acc:
        #     print('best model find at epoch', epoch)
        # best_acc = max(valid_accuracy, best_acc)
        # torch.save(model.state_dict(), os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch)))