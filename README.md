# Abstrack

This repository is the implementation of predict model for fx using RL and GAN.

# Setup

To create the container(here, we assume to use GPU)
``` 
docker run --gpus all -it -v $PWD:/work --workdir /work --name ganrl nvcr.io/nvidia/pytorch:21.09-py3   
```

To setup the conda environment
(kirl and kifin are module I create. in detail, please look at my repositories.)
``` 
conda init
source ~/.bashrc
conda create -n ganrl --clone base
conda activate ganrl

apt-get install -y git 
pip install signatory==1.2.1.1.4.0
pip install git+https://github.com/koyaishikawa/kirl
pip install git+https://github.com/koyaishikawa/kifin
```