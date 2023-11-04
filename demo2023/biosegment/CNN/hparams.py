"""
Size of audio window for which MFCC coefficients are calculated
"""
ANALYSIS_WINDOW = 0.03 # 0.02 second = 20 ms

"""
Step of ANALYSIS_WINDOW 
"""
ANALYSIS_STEP = 0.015 # 0.01 second overlap between consecutive windows

"""
number of vectors around the current vector to return.
This will cause VectorDataSource.read() method to return
a sequence of (SCOPE_LENGTH * 2 + 1) vectors (if enough
data is available), with the current vetor in the middle
"""
SCOPE_LENGTH = 10

"""
Feature configuration
Feature type: lfcc/mfcc
"""
FEATURE = 'lfcc'

"""
Number of Mel filters
"""
MEL_FILTERS = 40

"""
Number of MFCC coefficients to keep
"""
N_MFCC = 16

"""
Sampling rate of audio data
"""
SAMPLING_RATE = 16000

"""
ANALYSIS_WINDOW and ANALYSIS_STEP as number of samples
"""
BLOCK_SIZE = int(SAMPLING_RATE * ANALYSIS_WINDOW)
HOP_SIZE = int(SAMPLING_RATE * ANALYSIS_STEP)

"""
Compute delta and delta-delta of MFCC coefficients ?
"""
DELTA_1 = True
DELTA_2 = True

"""
Where to find data
"""
DATA_PATH = './data'

"""
Labels
"""
BIOTYPE = {
    "silence":0,
    "breath":1,
    "speech":2
}
