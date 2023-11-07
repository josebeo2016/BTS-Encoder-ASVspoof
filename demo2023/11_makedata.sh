#!/bin/bash

DF2021_DIR=/datab/Dataset/ASVspoof/LA/ASVspoof2021_DF_eval/
N=10000
# This script is used to copy smaller set of DF2021
mkdir DATA
mkdir DATA/flac

# Copy the flac files

find $DF2021_DIR/flac/ -type f -name "*.flac" | shuf -n $N | xargs -I {} cp {} DATA/flac/
ls DATA/flac/ | wc -l

# make protocol files
cd DATA/flac/
for file in *; do [[ -f "$file" ]] && echo "${file%.*}"; done >> ../protocol.txt

echo "Finished copying flac files and making protocol files"
echo "data path: DATA/flac/"
