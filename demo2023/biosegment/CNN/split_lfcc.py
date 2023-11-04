import pandas as pd
import numpy as np
import torch
import sys
import os
import yaml

# python split_lfcc.py config.yaml out/feats.h5 out/
if len(sys.argv) < 4:
    print("Usage: python {} config.yaml out/feats.h5 out/".format(sys.argv[0]))
    exit(1)
feats = pd.read_hdf(sys.argv[2])
print(feats)
new_feats = pd.DataFrame(columns=['fid', 'features', 'class', 'utt'])
# load config
config = yaml.load(open(sys.argv[1], 'r'), Loader=yaml.FullLoader)
feat_len = config['model']['feat_len']
print(feats.columns)
for i in feats.index:
    feat = feats['features'][i]
    print(feat.shape)
    # feat shape: (time, 60)
    if feat.shape[0] > feat_len:
        # devide into segments
        for j in range(0, feat.shape[0], feat_len):
            if j + feat_len > feat.shape[0]:
                continue
            new_feats = new_feats.append({'fid': feats['utterance-id'][i] + str(j), 'features': feat[j:j+feat_len, :], 'class': feats['speaker-id'][i], 'utt': feats['recording-id'][i]}, ignore_index=True)

    else:
        new_feats = new_feats.append({'fid': feats['utterance-id'][i], 'features': feat, 'class': feats['speaker-id'][i], 'utt': feats['recording-id'][i]}, ignore_index=True)
        

print(new_feats)
print("Saving to {}".format(os.path.join(sys.argv[3], 'feats_split.h5')))
new_feats.to_hdf(os.path.join(sys.argv[3], 'feats_split.h5'), key='feats', mode='w')