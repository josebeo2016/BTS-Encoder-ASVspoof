#!/bin/bash

# usage: bash 01_download_pretrained.sh
#
# URL of the file that needs to be downloaded
##########################################
# Download w2v model
##########################################
echo "Downloading xlsr2_300m pretrained model"
url="https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt"

# The folder where the downloaded file should be stored
directory="pretrained/"
filename="xlsr2_300m.pt"
# Check if the directory exists. If not, create the directory
if [ ! -d "$directory" ]; then
  mkdir -p "$directory"
fi

# Move to the directory
cd "$directory"

# check if the file exists. If yes, pass
if [ -f "$filename" ]; then
  echo "$filename exists."

else
  # Download the file
  wget "$url"
fi
cd ../
##########################################
# Download feats
##########################################
echo "Downloading pre-calculated BTS embedding for DF2021"
url="1gvsDc9PhCgzFwV2AYs7CsiSrbFLDb2Mm"
directory="feats/"
filename="feats_eval_2021.tar.gz"
# Check if the directory exists. If not, create the directory
if [ ! -d "$directory" ]; then
  mkdir -p "$directory"
fi

# Move to the directory
cd "$directory"

# check if the file exists. If yes, pass
if [ -f "$filename" ]; then
  echo "$filename exists."
else
  # Download the file
  gdown "$url"
  tar -xvf "$filename"
fi
cd ../
##########################################
# Download pretrained BTS-E model
##########################################
directory="pretrained/"
filename="full_trans64_concat.pth"
url="1L9TPpFXKafH7PTaWxTKA8sjQLSP5NyVz"
# Check if the directory exists. If not, create the directory
if [ ! -d "$directory" ]; then
  mkdir -p "$directory"
fi

# Move to the directory
cd "$directory"

# check if the file exists. If yes, pass
if [ -f "$filename" ]; then
  echo "$filename exists."
else
  # Download the file
  gdown "$url"
fi