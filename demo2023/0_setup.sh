#!/bin/bash
# Install dependency for fairseq

# Name of the conda environment
ENVNAME=fairseq

eval "$(conda shell.bash hook)"
conda activate ${ENVNAME}
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Install conda environment ${ENVNAME}"
    
    # conda env
    conda create -n ${ENVNAME} python=3.7 pip --yes
    conda activate ${ENVNAME}

    # install pytorch
    echo "===========Install pytorch==========="
    # conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

    # git clone fairseq
    #  fairseq 0.10.2 on pip does not work
    # git clone https://github.com/pytorch/fairseq
    # cd fairseq
    cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
    #  checkout this specific commit. Latest commit does not work
    # git checkout 862efab86f649c04ea31545ce28d13c59560113d
    pip install --editable ./

    # install scipy
    pip install scipy==1.7.3

    # install pandas
    pip install pandas==1.3.5

    # install protobuf
    pip install protobuf==3.20.3

    # install tensorboard
    pip install tensorboard==2.6.0
    pip install tensorboardX==2.6

    # install librosa
    pip install librosa==0.10.0

    # install pydub
    pip install pydub==0.25.1

    # install yaml
    pip install pyyaml

    # install tqdm
    pip install tqdm

    # install scikit-learn
    pip install scikit-learn==1.0.2
    # install h5py
    pip install h5py==3.7.0

    # install matplotlib
    pip install matplotlib

    # install spafe
    pip install spafe==0.1.2

    # install auditok
    pip install auditok==0.2.0

    # install gdown
    pip install gdown
    

else
    echo "Conda environment ${ENVNAME} has been installed"
fi

