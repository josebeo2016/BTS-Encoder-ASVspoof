#!/bin/bash
# 3_scoring.sh

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