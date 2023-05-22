from .preprocess import extract_lfcc, extract_mfcc
from .GMM_breath import GMMClassifier, ClassifierValidator, VectorDataSource
from .hparams import *
import pickle
from auditok import DataValidator, ADSFactory, DataSource, StreamTokenizer, BufferAudioSource, player_for
import soundfile as sf
import os


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# if __name__ == '__main__':
    
    # Load model
models = {}
for cls in ["breath", "silence", "speech"]:
    fp = open(BASE_DIR+"/out/{}.gmm".format(cls), "rb")
    models[cls]=pickle.load(fp)
    fp.close()
biotype = {
    "silence":0,
    "breath":1,
    "speech":2
}
gmm_classifier = GMMClassifier(models)

silence_validator = ClassifierValidator(gmm_classifier, "silence")
speech_validator = ClassifierValidator(gmm_classifier, "speech")
breath_validator = ClassifierValidator(gmm_classifier, "breath")
# Tokennizer
analysis_window_per_second = 1. / ANALYSIS_STEP

min_seg_length = 0.2 # second, min length of an accepted audio segment
max_seg_length = 10 # seconds, max length of an accepted audio segment
max_silence = 0.3 # second, max length tolerated of tolerated continuous signal that's not from the same class

tokenizer = StreamTokenizer(validator=breath_validator, min_length=int(min_seg_length * analysis_window_per_second),
                                    max_length=int(max_seg_length * analysis_window_per_second),
                                    max_continuous_silence= max_silence * analysis_window_per_second,
                                    mode = StreamTokenizer.DROP_TRAILING_SILENCE)
    
# Load data

def wav2bio(data, sr):
    # data, sr = sf.read(wav_path)
    lfcc = VectorDataSource(data=extract_lfcc(sig=data,sr=sr),scope=15)   
    # tokenized:
    lfcc.rewind()
    data = lfcc.read()
    # print(data)
    result = []
    while (data is not None):
        result.append(biotype[gmm_classifier.predict(data)[0][0]])
        data = lfcc.read()
    return result

# wav_path = "/root/dataset/ASVspoof/LA/ASVspoof2019_LA_train/flac/LA_T_5450704.flac"
# data, sr = sf.read(wav_path)
# wav2bio(data, sr)