#!/bin/bash
# usage: bash 2_evaldf.sh
# Enter conda environment

DATA_PATH="/datab/Dataset/ASVspoof/LA/" # path to downloaded DF dataset
BATCH_SIZE=10 # batch size for evaluation on GPU 24gb memory
MODEL_PATH="pretrained/tts_only_trans64_concat.pth" # path to the model
EVAL_OUTPUT="test.txt" # output file name
TRACK="DF"
SUBSET="eval" # eval or hidden_track or progress

eval "$(conda shell.bash hook)"

conda activate fairseq
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Cannot load fairseq, please run 0_setup.sh first"
    exit 1
fi

echo "Calculating detection score"
# if $EVAL_OUTPUT already exists, skip this step
if [ -e $EVAL_OUTPUT ]
then
    echo "$EVAL_OUTPUT exists, skipping this step"
else
    com="python main.py 
        --config configs/model_config_RawNet_Trans_64concat.yaml 
        --database_path $DATA_PATH
        --protocols_path $DATA_PATH
        --batch_size $BATCH_SIZE
        --eval_2021 
        --is_eval 
        --model_path $MODEL_PATH
        --eval_output $EVAL_OUTPUT
        --track DF";
    echo $com;
    eval $com;
fi
echo "Finished calculating detection score"

echo "Calculating EER"
cd eval-package/
com="python main.py 
    --cm-score-file ../$EVAL_OUTPUT
    --track $TRACK 
    --subset $SUBSET 
    --metadata keys"

echo $com
eval $com