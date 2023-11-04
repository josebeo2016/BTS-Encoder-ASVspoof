from json.tool import main
import wave
import pickle
import numpy as np
import pandas as pd
import argparse
from sklearn import mixture
from numpy import log, exp, infty, zeros_like, vstack, zeros, errstate, finfo, sqrt, floor, tile, concatenate, arange, meshgrid, ceil, linspace
from scipy.signal import lfilter
from .hparams import *
import soundfile as sf
import logging
from .LFCC_pipeline import lfcc

import librosa
from auditok import DataValidator, ADSFactory, DataSource, StreamTokenizer, BufferAudioSource, player_for
import h5py

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


class GMMClassifier():
    
    def __init__(self, models):
        """
        models is a dictionary: {"class_of_sound" : GMM_model_for_that_class, ...}
        """        
        self.models = models
    
    
    def predict(self, data):
        
        result = []
        
        for cls in self.models:
            
            llk = self.models[cls].score_samples(data)[0]
            llk = np.sum(llk)
            # print("{} {}".format(cls,llk))
            if len(self.models) > 1:
                result.append((cls, llk)) 
            else:
                if llk<-100:
                    result.append(("silence",0))
                else:
                    result.append((cls, llk))
        
        """
        return classification result as a sorted list of tuples ("class_of_sound", log_likelihood)
        best class is the first element in the list
        """
        
        return sorted(result, key=lambda f: - f[1])

            
    
class ClassifierValidator(DataValidator):
    
    def __init__(self, classifier, target):
        """
        classifier: a GMMClassifier object
        target: string
        """
        self.classifier = classifier
        self.target = target
        
    def is_valid(self, data):
        
        r = self.classifier.predict(data)
        # if (r[0][0] == self.target):
        # print(r[0][0],r[0][1], self.target)
        return r[0][0] == self.target
    

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
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='extract_feats.py',
                                     description='Extract log-mel spectrogram features.')

    parser.add_argument('feat_dir', type=str,
                        help='a path to ')
    
    parser.add_argument('model_dir', type=str,
                        help='a path to an')
    args = parser.parse_args()
    # Load feats
    feats = pd.read_hdf(f'{args.feat_dir}/feats.h5')
    training_data = {
        "breath": [],
        "silence": [],
        "speech": []
    }
    models = {}
    print(feats.columns)
    for i in feats.index:
        for cls in training_data:
            if cls == feats['speaker-id'][i]:
                training_data[cls].append(feats['features'][i])
        print("++++{} - {} - {}++++".format(feats['utterance-id'][i], feats['speaker-id'][i],feats['recording-id'][i]))

    for cls in training_data:
        data = training_data[cls]
        X = vstack(data)
        logging.info('+++++++++GMM {} start fitting'.format(cls))
        mod = mixture.GaussianMixture(n_components=NCOMP,
                              random_state=None,
                              covariance_type='diag',
                              max_iter=100,
                              verbose=2,
                              verbose_interval=1).fit(X)
        models[cls] = mod
        logging.info('+++++++++GMM init done - llh: %.5f+++++++++' % mod.lower_bound_)
    # # Train GMM
    
    # breath_gmm = mixture.GaussianMixture(n_components=NCOMP,
    #                           random_state=None,
    #                           covariance_type='diag',
    #                           max_iter=50,
    #                           verbose=2,
    #                           verbose_interval=1).fit(X)
    
    # logging.info('GMM init done - llh: %.5f' % breath_gmm.lower_bound_)
    # # Future: more biological sound
    # models = {}
    # # Save model
    # fp = open(f'{args.feat_dir}/breath.pkl', "wb")
    # pickle.dump(breath_gmm, fp, pickle.HIGHEST_PROTOCOL)
    # fp.close()
    for cls in training_data:
        fp = open("{}/{}.gmm".format(args.feat_dir,cls), "wb")
        pickle.dump(models[cls], fp, pickle.HIGHEST_PROTOCOL)
        fp.close()
    
    
    
    
    