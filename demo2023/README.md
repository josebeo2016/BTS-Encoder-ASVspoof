# Demo system wav2vec-bts-e

## Directory tree
- biosegment: The library to calculate BTS encoding. The see `__init__.py` for more detail
- core_scripts: Do not need to care.
- pretrained: The trained models are stored here, including the BTS-E and wav2vec 2.0
- eval-package: The library to calculate EER and t-DCF

## Dataset
- The DF2021 should be download [here](https://zenodo.org/records/4835108). Take long time to download
- The pre-calculated BTS encoding and pretrained model should be download [here](https://drive.google.com/drive/folders/1uuPaP2c117h6yWvNIyh-91JlMPnTQfxw?usp=drive_link)
- the script `1_download.sh` only supports downloading pretrained models.

## Installation
- Install Anaconda following the guideline in [https://conda.io/projects/conda/en/latest/user-guide/install/linux.html#](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html#)

## Demo scenario
- Setup environment `bash 0_setup.sh`
- Download pretrained: `bash 1_download.sh`
- Evaluate DF2021: `bash 2_evaldf.sh`
<!-- - Calculate EER: `bash 3_scoring.sh` -->

## Manual

```
CUDA_VISIBLE_DEVICES=2 python main.py --config configs/model_config_RawNet_Trans_64concat.yaml --batch_size 10 --eval_2021 --is_eval --model_path models/wav2vec-bio-trans64-concat/epoch_21.pth --eval_output ../../../docs/bio_tts_only_train_res/DF2021_wav2vec-bio-trans64-concat_epoch_21.txt --track DF
```

