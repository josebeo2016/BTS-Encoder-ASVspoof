#!/bin/bash
# 3_scoring.sh
eval "$(conda shell.bash hook)"

conda activate fairseq
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Cannot load fairseq, please run 0_setup.sh first"
    exit 1
fi
EVAL_OUTPUT=$1
TRACK="DF"
SUBSET="eval"

echo "Calculating EER"
cd eval-package/
pwd
com="python main.py 
    --cm-score-file $EVAL_OUTPUT
    --track $TRACK 
    --subset $SUBSET 
    --metadata keys"

echo $com
eval $com