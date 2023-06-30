from audioop import bias
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from torch.utils import data
from collections import OrderedDict
from torch.nn.parameter import Parameter
import math

# from . import biosegment
# from . import transformer
# from . import commons
# from . import cnns2s
import cnns2s
import transformer
import biosegment
import commons

___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

__modifier__ = "Phucdt"

def Mask_Generate(lengths, max_lengths= None, dtype= torch.float):
        '''
        lengths: [Batch]
        '''
        mask = torch.arange(max_lengths or torch.max(lengths))[None, :].to(lengths.device) < lengths[:, None]    # [Batch, Time]
        return mask.unsqueeze(1).to(dtype)  # [Batch, 1, Time]

class BioEmbedding(nn.Module):
    def __init__(self, device, num_bios, out_channels) -> None:
        super(nn.Module, self).__init__()
        self.out_channels = out_channels
        self.num_embeddings = num_bios
        self.emb = nn.Embedding(
            num_embeddings=num_bios,
            embedding_dim=out_channels
        )
        
    
    def forward(self, x):
        """

        Args:
            x (tensor): Biological sound segmentation vector [B, T_bio]


        Returns:
            _type_: _description_ 
        """
        x = self.emb(x).transpose(2,1)# [B, out_channels, T_bio]
        
        

class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)


    def __init__(self, device,out_channels, kernel_size,in_channels=1,sample_rate=16000,
                 stride=1, padding=0, dilation=1, bias=False, groups=1):

        super(SincConv,self).__init__()

        if in_channels != 1:
            
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)
        
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate=sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1

        self.device=device   
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')
        
        
        # initialize filterbanks using Mel scale
        NFFT = 512
        f=int(self.sample_rate/2)*np.linspace(0,1,int(NFFT/2)+1)
        fmel=self.to_mel(f)   # Hz to mel conversion
        fmelmax=np.max(fmel)
        fmelmin=np.min(fmel)
        filbandwidthsmel=np.linspace(fmelmin,fmelmax,self.out_channels+1)
        filbandwidthsf=self.to_hz(filbandwidthsmel)  # Mel to Hz conversion
        self.mel=filbandwidthsf
        self.hsupp=torch.arange(-(self.kernel_size-1)/2, (self.kernel_size-1)/2+1)
        self.band_pass=torch.zeros(self.out_channels,self.kernel_size)
    
       
        
    def forward(self,x):
        for i in range(len(self.mel)-1):
            fmin=self.mel[i]
            fmax=self.mel[i+1]
            hHigh=(2*fmax/self.sample_rate)*np.sinc(2*fmax*self.hsupp/self.sample_rate)
            hLow=(2*fmin/self.sample_rate)*np.sinc(2*fmin*self.hsupp/self.sample_rate)
            hideal=hHigh-hLow
            
            self.band_pass[i,:]=Tensor(np.hamming(self.kernel_size))*Tensor(hideal)
        
        band_pass_filter=self.band_pass.to(self.device)

        self.filters = (band_pass_filter).view(self.out_channels, 1, self.kernel_size)
        # print(x)
        return F.conv1d(x, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)


        
class Residual_block(nn.Module):
    def __init__(self, nb_filts, first = False):
        super(Residual_block, self).__init__()
        self.first = first
        
        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features = nb_filts[0])
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        
        self.conv1 = nn.Conv1d(in_channels = nb_filts[0],
			out_channels = nb_filts[1],
			kernel_size = 3,
			padding = 1,
			stride = 1)
        
        self.bn2 = nn.BatchNorm1d(num_features = nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels = nb_filts[1],
			out_channels = nb_filts[1],
			padding = 1,
			kernel_size = 3,
			stride = 1)
        
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels = nb_filts[0],
				out_channels = nb_filts[1],
				padding = 0,
				kernel_size = 1,
				stride = 1)
            
        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)
        
    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x
            
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        
        if self.downsample:
            identity = self.conv_downsample(identity)
            
        out += identity
        out = self.mp(out)
        return out

# https://github.com/bentrevett/pytorch-seq2seq/
class bioEncoderRNN(nn.Module):
    def __init__(self, d_args, device) -> None:
        super(bioEncoderRNN, self).__init__()

        self.device=device
        
        self.bio_emb = nn.Embedding(d_args['n_bios'], d_args['bio_dim'])
        self.bio_dim = d_args['bio_dim'] 

        # nn.init.normal_(self.bio_emb.weight, 0.0, d_args['bio_dim']**-0.5)
        
        # length scoring == # fc1 out features
        self.rnn = nn.GRU(d_args['bio_dim'], d_args['bio_rnn'], 1, batch_first=True)
        
        
        self.bio_scoring = nn.Linear(in_features = d_args['bio_rnn'],
			out_features = d_args['nb_fc_node'],bias=True)
        
    def forward(self, bio, bio_lengths):
        
        bio = self.bio_emb(bio) # [b, bio_length, bio_dim]
        # print(bio.size())
        bio_lengths = bio_lengths.cpu().numpy()
 
        bio = nn.utils.rnn.pack_padded_sequence(
                        bio, bio_lengths, batch_first=True)

        self.rnn.flatten_parameters()
        bio, hidden = self.rnn(bio)
        bio, _ = nn.utils.rnn.pad_packed_sequence(
            bio, batch_first=True)
        
        # hidden [b, bio_dim]
        bio_scoring = self.bio_scoring(hidden[-1,:,:])
        # bio_scoring = torch.tanh(self.bio_scoring(hidden[-1,:,:]))
        return bio_scoring

class bioEncoderRNNsmall(nn.Module):
    def __init__(self, d_args, device) -> None:
        super(bioEncoderRNNsmall, self).__init__()

        self.device=device
        
        self.bio_emb = nn.Embedding(d_args['n_bios'], d_args['bio_dim'])
        self.bio_dim = d_args['bio_dim'] 

        # nn.init.normal_(self.bio_emb.weight, 0.0, d_args['bio_dim']**-0.5)
        
        # length scoring == # fc1 out features
        self.rnn = nn.GRU(d_args['bio_dim'], d_args['bio_rnn'], 1, batch_first=True)
        
        
        self.bio_scoring = nn.Linear(in_features = d_args['bio_rnn'],
			out_features = d_args['bio_out'],bias=True)
        
    def forward(self, bio, bio_lengths):
        
        bio = self.bio_emb(bio) # [b, bio_length, bio_dim]
        # print(bio.size())
        bio_lengths = bio_lengths.cpu().numpy()
 
        bio = nn.utils.rnn.pack_padded_sequence(
                        bio, bio_lengths, batch_first=True)

        self.rnn.flatten_parameters()
        bio, hidden = self.rnn(bio)
        bio, _ = nn.utils.rnn.pad_packed_sequence(
            bio, batch_first=True)
        
        # last hidden [b, bio_dim]
        bio_scoring = self.bio_scoring(hidden[-1,:,:])
        # bio_scoring = torch.tanh(self.bio_scoring(hidden[-1,:,:]))
        return bio_scoring


class bioEncoderConv(nn.Module):
    def __init__(self, d_args, device) -> None:
        super(bioEncoderConv, self).__init__()

        self.device=device       
        self.conv = cnns2s.Encoder(d_args['n_bios'],
                                   d_args['bio_dim'],
                                   d_args['bio_hid'],
                                   d_args['n_layers'],
                                   device=device
                                   )
        self.bio_scoring = nn.Linear(in_features = d_args['bio_dim'],
			out_features = d_args['bio_out'],bias=True)
        
    def forward(self, bio, bio_lengths):
        
        bio, _ = self.conv(bio)
        # print(bio.size())
        bio = bio[:,-1,:]
        bio_scoring = self.bio_scoring(bio)
        # print(bio_scoring.size())
        return bio_scoring


class bioEncoderTransformer(nn.Module):
    def __init__(self, d_args, device):
        super(bioEncoderTransformer, self).__init__()

        self.device=device
        self.bio_dim = d_args['bio_dim']
        self.bio_embedding = nn.Embedding(d_args['n_bios'], d_args['bio_dim'])
        nn.init.normal_(self.bio_embedding.weight, 0.0, d_args['bio_dim']**-0.5)

        self.encoder = transformer.Encoder(
                        d_args['bio_dim'],
                        d_args['pf_dim'],
                        d_args['n_heads'],
                        d_args['n_layers'],
                        )
        # self.bio_scoring = nn.Linear(in_features = d_args['bio_rnn'],
        #         out_features = d_args['nb_fc_node'],bias=True)
        self.bio_scoring= nn.Conv1d(d_args['bio_dim'], d_args['bio_out'], 1)
    
    def forward(self, bio, bio_lengths):
        bio = self.bio_embedding(bio) * math.sqrt(self.bio_dim) # [b, bio_lengths, bio_dim]
        bio = torch.transpose(bio, 1, -1) # [b, bio_dim, bio_lengths]
        bio_mask = torch.unsqueeze(commons.sequence_mask(bio_lengths, bio.size(2)), 1).to(bio.dtype)

        bio = self.encoder(bio * bio_mask, bio_mask) # [b, bio_dim, bio_lengths]

        bio_scoring = self.bio_scoring(bio) * bio_mask

        return bio_scoring[:,:,-1] # [b, nb_fc_node]
        # return bio_scoring # for gru


class bioEncoderTransformersmall(nn.Module):
    def __init__(self, d_args, device):
        super(bioEncoderTransformersmall, self).__init__()

        self.device=device
        self.bio_dim = d_args['bio_dim']
        self.bio_embedding = nn.Embedding(d_args['n_bios'], d_args['bio_dim'])
        nn.init.normal_(self.bio_embedding.weight, 0.0, d_args['bio_dim']**-0.5)

        self.encoder = transformer.Encoder(
                        d_args['bio_dim'],
                        d_args['pf_dim'],
                        d_args['n_heads'],
                        d_args['n_layers'],
                        )
        # self.bio_scoring = nn.Linear(in_features = d_args['bio_rnn'],
        #         out_features = d_args['nb_fc_node'],bias=True)
        self.bio_scoring= nn.Conv1d(d_args['bio_dim'], d_args['bio_out'], 1)
    
    def forward(self, bio, bio_lengths):
        bio = self.bio_embedding(bio) * math.sqrt(self.bio_dim) # [b, bio_lengths, bio_dim]
        bio = torch.transpose(bio, 1, -1) # [b, bio_dim, bio_lengths]
        bio_mask = torch.unsqueeze(commons.sequence_mask(bio_lengths, bio.size(2)), 1).to(bio.dtype)

        bio = self.encoder(bio * bio_mask, bio_mask) # [b, bio_dim, bio_lengths]

        bio_scoring = self.bio_scoring(bio) * bio_mask

        return bio_scoring[:,:,-1] # [b, bio_out]


class bioEncoderlight(nn.Module):
    def __init__(self, d_args, device):
        super(bioEncoderlight, self).__init__()

        self.device=device
        # self.bio_dim = d_args['bio_dim']
        self.bio_embedding = nn.Embedding(d_args['n_bios'], d_args['bio_dim'])


        self.conv1 = nn.Conv1d(d_args['bio_dim'], 256, 1)
        self.conv2 = nn.Conv1d(256, 512, 1)
        
        self.fc1 = nn.Conv1d(512, d_args['nb_fc_node'], 1)
        # self.fc2 = nn.Linear(d_args['nb_fc_node'],d_args['nb_fc_node'], bias=True)
        
    
    def forward(self, bio, bio_lengths):
        bio = self.bio_embedding(bio) # [b, len, bio_dim]
        bio = torch.transpose(bio, 1, -1) # [b, bio_dim, len]
        bio = self.conv1(bio)
        bio = self.conv2(bio)
        bio = self.fc1(bio)
        # bio = self.fc2(bio)
        # bio_scoring = torch.transpose(bio_scoring, 1, -1) # [b, len, bio_dim]
        # print (bio.size())
        return bio[:,:,-1] # [b, nb_fc_node]
        

class RawNet_new(nn.Module):
    def __init__(self, d_args, device):
        super(RawNet_new, self).__init__()

        
        self.device=device
        #PHUCDT
        # self.bioScoring = bioEncoderConv(d_args, self.device)
        # self.bioScoring = bioEncoderRNN(d_args, self.device)
        self.bioScoring = bioEncoderTransformer(d_args, device)
        # self.bioScoring = bioEncoderlight(d_args, self.device)
        # self.bioScoring = bioEncoderRNNsmall(d_args, self.device)
        # self.bioScoring = bioEncoderTransformersmall(d_args, self.device)
        
        self.gru = nn.GRU(input_size = d_args['bio_out'],
			hidden_size = d_args['gru_node'],
			num_layers = d_args['nb_gru_layer'],
			batch_first = True)
        
        self.fc2_gru = nn.Linear(in_features = d_args['gru_node'],
			out_features = d_args['nb_classes'])
        			
        self.sig = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x, bio = None, bio_lengths = None, y = None):
        
        bio_scoring = self.bioScoring(bio, bio_lengths)
        
        bio_scoring = bio_scoring.permute(0, 2, 1)     #(batch, filt, time) >> (batch, time, filt)
        self.gru.flatten_parameters()
        bio_scoring, _ = self.gru(bio_scoring)
        bio_scoring = bio_scoring[:,-1,:]
        # print(bio_scoring.size())
        bio_scoring = self.fc2_gru(bio_scoring)
        output=self.logsoftmax(bio_scoring)
      
        return output, bio_scoring
        
        

    def _make_attention_fc(self, in_features, l_out_features):

        l_fc = []
        
        l_fc.append(nn.Linear(in_features = in_features,
			        out_features = l_out_features))

        

        return nn.Sequential(*l_fc)


    def _make_layer(self, nb_blocks, nb_filts, first = False):
        layers = []
        #def __init__(self, nb_filts, first = False):
        for i in range(nb_blocks):
            first = first if i == 0 else False
            layers.append(Residual_block(nb_filts = nb_filts,
				first = first))
            if i == 0: nb_filts[0] = nb_filts[1]
            
        return nn.Sequential(*layers)

    def summary(self, input_size, batch_size=-1, device="cuda", print_fn = None):
        if print_fn == None: printfn = print
        model = self
        
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)
                
                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
						[-1] + list(o.size())[1:] for o in output
					]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    if len(summary[m_key]["output_shape"]) != 0:
                        summary[m_key]["output_shape"][0] = batch_size
                        
                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params
                
            if (
				not isinstance(module, nn.Sequential)
				and not isinstance(module, nn.ModuleList)
				and not (module == model)
			):
                hooks.append(module.register_forward_hook(hook))
                
        device = device.lower()
        assert device in [
			"cuda",
			"cpu",
		], "Input device is not valid, please specify 'cuda' or 'cpu'"
        
        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        if isinstance(input_size, tuple):
            input_size = [input_size]
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
        summary = OrderedDict()
        hooks = []
        model.apply(register_hook)
        model(*x)
        for h in hooks:
            h.remove()
            
        print_fn("----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        print_fn(line_new)
        print_fn("================================================================")
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20}  {:>25} {:>15}".format(
				layer,
				str(summary[layer]["output_shape"]),
				"{0:,}".format(summary[layer]["nb_params"]),
			)
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]
            print_fn(line_new)

# your task
def get_Bio(X_pad, fs):
    bio = biosegment.wav2bio(X_pad, fs)
    bio_inp = torch.IntTensor(bio)
    bio_length = torch.IntTensor([len(bio)])
    return bio_inp, bio_length

class RawNet(nn.Module):
    def __init__(self, d_args, device):
        super(RawNet, self).__init__()
        self.device=device

        self.Sinc_conv=SincConv(device=self.device,
			out_channels = d_args['filts'][0],
			kernel_size = d_args['first_conv'],
                        in_channels = d_args['in_channels']
        )
        
        self.first_bn = nn.BatchNorm1d(num_features = d_args['filts'][0])
        self.selu = nn.SELU(inplace=True)
        self.block0 = nn.Sequential(Residual_block(nb_filts = d_args['filts'][1], first = True))
        self.block1 = nn.Sequential(Residual_block(nb_filts = d_args['filts'][1]))
        self.block2 = nn.Sequential(Residual_block(nb_filts = d_args['filts'][2]))
        d_args['filts'][2][0] = d_args['filts'][2][1]
        self.block3 = nn.Sequential(Residual_block(nb_filts = d_args['filts'][2]))
        self.block4 = nn.Sequential(Residual_block(nb_filts = d_args['filts'][2]))
        self.block5 = nn.Sequential(Residual_block(nb_filts = d_args['filts'][2]))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc_attention0 = self._make_attention_fc(in_features = d_args['filts'][1][-1],
            l_out_features = d_args['filts'][1][-1])
        self.fc_attention1 = self._make_attention_fc(in_features = d_args['filts'][1][-1],
            l_out_features = d_args['filts'][1][-1])
        self.fc_attention2 = self._make_attention_fc(in_features = d_args['filts'][2][-1],
            l_out_features = d_args['filts'][2][-1])
        self.fc_attention3 = self._make_attention_fc(in_features = d_args['filts'][2][-1],
            l_out_features = d_args['filts'][2][-1])
        self.fc_attention4 = self._make_attention_fc(in_features = d_args['filts'][2][-1],
            l_out_features = d_args['filts'][2][-1])
        self.fc_attention5 = self._make_attention_fc(in_features = d_args['filts'][2][-1],
            l_out_features = d_args['filts'][2][-1])

        self.bn_before_gru = nn.BatchNorm1d(num_features = d_args['filts'][2][-1])
        self.gru = nn.GRU(input_size = d_args['filts'][2][-1],
			hidden_size = d_args['gru_node'],
			num_layers = d_args['nb_gru_layer'],
			batch_first = True)

        
        self.fc1_gru = nn.Linear(in_features = d_args['gru_node'],
			out_features = d_args['nb_fc_node'])
        
        #PHUCDT
        # self.bioScoring = bioEncoderConv(d_args, self.device)
        # self.bioScoring = bioEncoderRNN(d_args, self.device) #add
        # self.bioScoring = bioEncoderTransformer(d_args, device) #add
        # self.bioScoring = bioEncoderlight(d_args, self.device)
        # self.bioScoring = bioEncoderRNNsmall(d_args, self.device) #concat
        self.bioScoring = bioEncoderTransformersmall(d_args, self.device) #concat
        
        # self.fc1 = nn.Linear(in_features = d_args['nb_fc_node'],
		# 	out_features = d_args['bio_out'],bias=True)
        
        # Concat
        self.fc2_gru = nn.Linear(in_features = d_args['nb_fc_node']+d_args['bio_out'],
			out_features = d_args['nb_classes'],bias=True)
        
        # ADD
        # self.fc2_gru = nn.Linear(in_features = d_args['nb_fc_node'],
		# 	out_features = d_args['nb_classes'],bias=True)

        self.sig = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x, y = None):
        X_pad = x.squeeze(0).detach().cpu().numpy()
        bio, bio_lengths = get_Bio(X_pad, 16000)
        bio = bio.unsqueeze(0).to(self.device)
        bio_lengths = bio_lengths.to(self.device)

        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x=x.view(nb_samp,1,len_seq)
        
        x = self.Sinc_conv(x)    
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.first_bn(x)
        x = self.selu(x)
        
        x0 = self.block0(x)
        y0 = self.avgpool(x0).view(x0.size(0), -1) # torch.Size([batch, filter])
        y0 = self.fc_attention0(y0)
        y0 = self.sig(y0).view(y0.size(0), y0.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x0 * y0 + y0  # (batch, filter, time) x (batch, filter, 1)
        

        x1 = self.block1(x)
        y1 = self.avgpool(x1).view(x1.size(0), -1) # torch.Size([batch, filter])
        y1 = self.fc_attention1(y1)
        y1 = self.sig(y1).view(y1.size(0), y1.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x1 * y1 + y1 # (batch, filter, time) x (batch, filter, 1)

        x2 = self.block2(x)
        y2 = self.avgpool(x2).view(x2.size(0), -1) # torch.Size([batch, filter])
        y2 = self.fc_attention2(y2)
        y2 = self.sig(y2).view(y2.size(0), y2.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x2 * y2 + y2 # (batch, filter, time) x (batch, filter, 1)

        x3 = self.block3(x)
        y3 = self.avgpool(x3).view(x3.size(0), -1) # torch.Size([batch, filter])
        y3 = self.fc_attention3(y3)
        y3 = self.sig(y3).view(y3.size(0), y3.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x3 * y3 + y3 # (batch, filter, time) x (batch, filter, 1)

        x4 = self.block4(x)
        y4 = self.avgpool(x4).view(x4.size(0), -1) # torch.Size([batch, filter])
        y4 = self.fc_attention4(y4)
        y4 = self.sig(y4).view(y4.size(0), y4.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x4 * y4 + y4 # (batch, filter, time) x (batch, filter, 1)

        x5 = self.block5(x)
        y5 = self.avgpool(x5).view(x5.size(0), -1) # torch.Size([batch, filter])
        y5 = self.fc_attention5(y5)
        y5 = self.sig(y5).view(y5.size(0), y5.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x5 * y5 + y5 # (batch, filter, time) x (batch, filter, 1)

        x = self.bn_before_gru(x)
        x = self.selu(x)
        x = x.permute(0, 2, 1)     #(batch, filt, time) >> (batch, time, filt)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:,-1,:]
        x = self.fc1_gru(x)
        
        #PHUCDT
        if (bio is not None):
            bio_scoring = self.bioScoring(bio, bio_lengths)
            # x = x + bio_scoring # add the conditioning bio scoring 
            x = torch.cat((x, bio_scoring), 1) # concat the conditioning bio scoring 

        x = self.fc2_gru(x)
        
        output=self.logsoftmax(x)
      
        return output
        
        

    def _make_attention_fc(self, in_features, l_out_features):

        l_fc = []
        
        l_fc.append(nn.Linear(in_features = in_features,
			        out_features = l_out_features))

        

        return nn.Sequential(*l_fc)


    def _make_layer(self, nb_blocks, nb_filts, first = False):
        layers = []
        #def __init__(self, nb_filts, first = False):
        for i in range(nb_blocks):
            first = first if i == 0 else False
            layers.append(Residual_block(nb_filts = nb_filts,
				first = first))
            if i == 0: nb_filts[0] = nb_filts[1]
            
        return nn.Sequential(*layers)

    def summary(self, input_size, batch_size=-1, device="cuda", print_fn = None):
        if print_fn == None: printfn = print
        model = self
        
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)
                
                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
						[-1] + list(o.size())[1:] for o in output
					]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    if len(summary[m_key]["output_shape"]) != 0:
                        summary[m_key]["output_shape"][0] = batch_size
                        
                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params
                
            if (
				not isinstance(module, nn.Sequential)
				and not isinstance(module, nn.ModuleList)
				and not (module == model)
			):
                hooks.append(module.register_forward_hook(hook))
                
        device = device.lower()
        assert device in [
			"cuda",
			"cpu",
		], "Input device is not valid, please specify 'cuda' or 'cpu'"
        
        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        if isinstance(input_size, tuple):
            input_size = [input_size]
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
        summary = OrderedDict()
        hooks = []
        model.apply(register_hook)
        model(*x)
        for h in hooks:
            h.remove()
            
        print_fn("----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        print_fn(line_new)
        print_fn("================================================================")
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20}  {:>25} {:>15}".format(
				layer,
				str(summary[layer]["output_shape"]),
				"{0:,}".format(summary[layer]["nb_params"]),
			)
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]
            print_fn(line_new)
